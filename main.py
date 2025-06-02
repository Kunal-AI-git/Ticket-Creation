import json
import re
import pickle
import numpy as np
import faiss
import os
import shutil
import spacy
import requests
import validators
import logging
import uuid
import time
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Any
import mimetypes
from jira import JIRA
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, HTTPError

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Directory for file uploads (use absolute path)
UPLOAD_DIR = os.path.abspath("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.pdf'}

# File paths for FAISS index and metadata
INDEX_FILE = "tickets_index.faiss"
META_FILE = "ticket_metadata.pkl"
CONVERSATION_STATE_FILE = "conversation_state.pkl"

# Simulated session storage
SESSION_FINALIZED_TICKETS = {}

# Jira configuration
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
JIRA_ISSUE_TYPE = os.getenv("JIRA_ISSUE_TYPE", "Task")
JIRA_URL = f"https://{JIRA_DOMAIN}" if JIRA_DOMAIN else None

# Component to role mapping for assignee logic
COMPONENT_ROLE_MAPPING = {
    "authentication": ["Backend Developer", "Security Engineer"],
    "dashboard": ["Frontend Developer", "UI/UX Designer"],
    "payment gateway": ["Backend Developer", "Payment Specialist"],
    "user profile": ["Frontend Developer", "Backend Developer"],
    "notifications": ["Backend Developer", "DevOps Engineer"],
    "search": ["Backend Developer", "Data Engineer"]
}

# Default assignee (account ID or email) if no suitable assignee is found
DEFAULT_ASSIGNEE = os.getenv("JIRA_DEFAULT_ASSIGNEE", None)

# Debug environment variables
logger.info(f"JIRA_URL: {JIRA_URL}")
logger.info(f"JIRA_EMAIL: {JIRA_EMAIL}")
logger.info(f"JIRA_API_TOKEN: {JIRA_API_TOKEN[:4] + '...' if JIRA_API_TOKEN else None}")
logger.info(f"JIRA_PROJECT_KEY: {JIRA_PROJECT_KEY}")
logger.info(f"JIRA_ISSUE_TYPE: {JIRA_ISSUE_TYPE}")
logger.info(f"JIRA_DEFAULT_ASSIGNEE: {DEFAULT_ASSIGNEE}")

# Initialize Jira client
def init_jira_client(retries=3, delay=5):
    if not all([JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY]):
        logger.error("Missing required Jira configuration in .env file")
        raise RuntimeError("Jira configuration incomplete")
    
    for attempt in range(retries):
        try:
            client = JIRA(
                server=JIRA_URL,
                basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN),
                options={"verify": True}
            )
            server_info = client.server_info()
            logger.info(f"Connected to Jira. Server version: {server_info['version']}")
            return client
        except (ConnectionError, HTTPError) as e:
            logger.warning(f"Jira connection attempt {attempt + 1}/{retries} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise RuntimeError(f"Jira connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error initializing Jira client: {str(e)}")
            raise RuntimeError(f"Jira initialization failed: {str(e)}")

try:
    jira_client = init_jira_client()
    logger.info("Jira client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Jira client: {str(e)}")
    raise RuntimeError(f"Jira initialization failed: {str(e)}")

# ---- Models ----
class Ticket(BaseModel):
    id: str
    title: str
    issue: str
    priority: str
    component: str
    description: Optional[str] = None
    resolution: Optional[str] = None
    attachments: Optional[List[Dict]] = None
    issue_type: Optional[str] = None

class UserInput(BaseModel):
    user_input: str
    conversation_id: Optional[str] = None

class UploadInput(BaseModel):
    conversation_id: str
    link: Optional[str] = None

class AnalysisResult(BaseModel):
    status: str
    missing_fields: list
    quality_issues: list

class SearchRequest(BaseModel):
    query: str

# ---- State Persistence Functions ----
def save_conversation_state(state: Dict):
    try:
        if os.path.exists(CONVERSATION_STATE_FILE) and not os.access(CONVERSATION_STATE_FILE, os.W_OK):
            raise PermissionError(f"No write permission for {CONVERSATION_STATE_FILE}")
        with open(CONVERSATION_STATE_FILE, "wb") as f:
            pickle.dump(state, f)
        logger.info("Conversation state saved to disk")
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to save conversation state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save conversation state: {str(e)}")

def load_conversation_state() -> Dict:
    try:
        if os.path.exists(CONVERSATION_STATE_FILE):
            with open(CONVERSATION_STATE_FILE, "rb") as f:
                state = pickle.load(f)
                if not isinstance(state, dict):
                    logger.error("Loaded conversation state is not a dictionary")
                    return {}
                for cid, s in state.items():
                    if not isinstance(s, dict):
                        logger.warning(f"Invalid state for {cid}, reinitializing")
                        state[cid] = {
                            "mode": "general",
                            "ticket_info": {},
                            "last_response": None,
                            "context_history": []
                        }
                        continue
                    if not isinstance(s.get("context_history", []), list):
                        logger.warning(f"Invalid context_history for {cid}, reinitializing")
                        s["context_history"] = []
                    if "mode" not in s:
                        logger.warning(f"Missing mode for {cid}, setting to general")
                        s["mode"] = "general"
                    if "ticket_info" not in s:
                        s["ticket_info"] = {}
                    if "last_response" not in s:
                        s["last_response"] = None
                logger.info("Conversation state loaded from disk")
                return state
        return {}
    except Exception as e:
        logger.error(f"Failed to load conversation state: {str(e)}")
        return {}

# ---- Mock RAG Chatbot ----
class MockRAGChatbot:
    def _init_(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            logger.error(f"Failed to load spaCy model 'en_core_web_sm': {str(e)}")
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")

    def generate_response(self, user_input: str) -> Dict:
        doc = self.nlp(user_input)
        entities = [ent.text for ent in doc.ents]
        return {
            "response": f"Received your query: '{user_input}'. Detected entities: {entities}. How can I assist you further?"
        }

# ---- Ticket Analysis Agent ----
class TicketAnalysisAgent:
    def _init_(self):
        self.required_fields = ["title", "issue", "priority", "component"]
        self.priority_options = ["low", "medium", "high", "critical"]
        self.valid_components = ["authentication", "dashboard", "payment gateway", "user profile", "notifications", "search"]
        self.valid_issue_types = ["Task", "Bug"]
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            logger.error(f"Failed to load spaCy model 'en_core_web_sm': {str(e)}")
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")

    def analyze_ticket(self, ticket: Dict) -> Dict:
        missing_fields = []
        quality_issues = []

        for field in self.required_fields:
            if field not in ticket or not ticket[field]:
                missing_fields.append(field)

        if ticket.get("priority") and ticket["priority"].lower() not in self.priority_options:
            quality_issues.append(f"Invalid priority: {ticket['priority']}. Must be one of {self.priority_options}")
        if ticket.get("title") and len(ticket["title"]) < 5:
            quality_issues.append("Title is too short. Provide a descriptive title (min 5 characters).")
        if ticket.get("issue") and len(ticket["issue"]) < 20:
            quality_issues.append("Issue description is too brief. Provide more details (min 20 characters).")
        if ticket.get("component") and ticket["component"].lower() not in self.valid_components:
            quality_issues.append(f"Invalid component: {ticket['component']}. Must be one of {self.valid_components}")
        if ticket.get("issue_type") and ticket["issue_type"] not in self.valid_issue_types:
            quality_issues.append(f"Invalid issue type: {ticket['issue_type']}. Must be one of {self.valid_issue_types}")

        field_values = {field: ticket.get(field, "").strip().lower() for field in self.required_fields}
        for i, field1 in enumerate(self.required_fields):
            for field2 in self.required_fields[i + 1:]:
                if field_values[field1] and field_values[field2]:
                    if field1 == "priority" and field_values[field1] in self.priority_options:
                        continue
                    if field2 == "priority" and field_values[field2] in self.priority_options:
                        continue
                    if field_values[field1] == field_values[field2]:
                        flagged_field = self._choose_field_to_flag(field1, field2)
                        quality_issues.append(
                            f"{flagged_field.capitalize()} is identical to {field2 if flagged_field == field1 else field1}. "
                            f"Please provide a more distinct {flagged_field}."
                        )
                    else:
                        doc1 = self.nlp(field_values[field1])
                        doc2 = self.nlp(field_values[field2])
                        similarity = doc1.similarity(doc2)
                        if similarity > 0.9:
                            flagged_field = self._choose_field_to_flag(field1, field2)
                            quality_issues.append(
                                f"{flagged_field.capitalize()} is too similar to {field2 if flagged_field == field1 else field1}. "
                                f"Please provide a more distinct {flagged_field}."
                            )

        if ticket.get("issue"):
            issue_doc = self.nlp(ticket["issue"].strip())
            meaningful_tokens = [token for token in issue_doc if not token.is_stop and not token.is_punct]
            if len(meaningful_tokens) < 5:
                quality_issues.append(
                    "Issue description lacks sufficient detail. Please include specific information such as steps to reproduce, "
                    "error messages, or affected features."
                )

        status = "success" if not missing_fields and not quality_issues else "incomplete"
        return {
            "status": status,
            "missing_fields": missing_fields,
            "quality_issues": quality_issues
        }

    def _choose_field_to_flag(self, field1: str, field2: str) -> str:
        priority = {"title": 1, "component": 2, "issue": 3, "priority": 4}
        return field1 if priority.get(field1, 4) < priority.get(field2, 4) else field2

    def generate_guidance(self, analysis: Dict) -> Dict:
        guidance = []
        for field in analysis["missing_fields"]:
            guidance.append(f"Please provide the '{field}' field. It's required for ticket submission.")
        for issue in analysis["quality_issues"]:
            guidance.append(issue)
        return {"guidance": guidance}

    def summarize_resolutions(self, similar_tickets: List[Dict]) -> str:
        resolutions = [ticket.get("resolution", "").strip() for ticket in similar_tickets if ticket.get("resolution", "").strip()]
        if not resolutions:
            return "No resolution information available for similar tickets."
        combined = " ".join(resolutions)
        doc = self.nlp(combined)
        summary = " ".join([sent.text for sent in doc.sents][:2])
        return summary or "Unable to generate summary from resolutions."

    def execute(self, method: str, data: Dict) -> Dict:
        if method == "analyze_ticket":
            return self.analyze_ticket(data)
        elif method == "generate_guidance":
            return self.generate_guidance(data)
        elif method == "summarize_resolutions":
            return {"summarized_resolution": self.summarize_resolutions(data)}
        else:
            raise ValueError(f"Unknown method: {method}")

# ---- Ticket Creation Agent ----
class TicketCreationAgent:
    def _init_(self, ticket_creation_api_url: str, ticket_analyzer: 'TicketAnalysisAgent'):
        self.ticket_creation_api_url = ticket_creation_api_url
        self.ticket_analyzer = ticket_analyzer
        self.required_fields = ["title", "issue", "priority", "component"]
        self.conversation_state = load_conversation_state()
        self.attachments = {}
        self.attachments_metadata = {}
        self.finalized_tickets = SESSION_FINALIZED_TICKETS
        self.valid_priorities = ["low", "medium", "high", "critical"]
        self.valid_components = ["authentication", "dashboard", "payment gateway", "user profile", "notifications", "search"]
        self.valid_issue_types = ["Task", "Bug"]
        self.issue_phrases = [
            "not loading", "not working", "issue with", "problem with", "bug", 
            "incorrect results", "error", "broken", "fails", "issue", "crash", 
            "down", "slow", "missing", "failed", "blocker"
        ]
        self.bug_keywords = [
            "error", "crash", "not working", "bug", "issue", "fail", "failed", "broken",
            "incorrect", "problem", "defect", "blocker", "exception", "glitch"
        ]
        self.task_keywords = [
            "add", "implement", "update", "enhance", "improve", "create", "new",
            "feature", "request", "change", "develop", "build"
        ]
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            logger.error(f"Failed to load spaCy model 'en_core_web_sm': {str(e)}")
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")

    def create_ticket(self, user_input: str, conversation_id: Optional[str] = None) -> Dict:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        conversation_id = str(conversation_id)
        if conversation_id not in self.conversation_state:
            self.start_conversation(conversation_id)
        response = self.continue_conversation(user_input, conversation_id)
        response["conversation_id"] = conversation_id
        save_conversation_state(self.conversation_state)
        return response

    def start_conversation(self, conversation_id: str) -> str:
        self.conversation_state[conversation_id] = {
            "upload_asked": False,
            "upload_done_asked": False,
            "current_field": "title",
            "stage": "intro",
            "fields_to_revise": [],
            "ticket_info": {},
            "field_order": ["title", "issue", "priority", "component", "issue_type"],
            "current_field_index": 0
        }

        self.attachments[conversation_id] = []
        self.attachments_metadata[conversation_id] = []
        logger.info(f"Started conversation: {conversation_id}")
        save_conversation_state(self.conversation_state)
        return "Hi! Let's create a ticket. Please provide a brief title for the issue."

    def continue_conversation(self, user_input: str, conversation_id: str) -> Dict:
        conversation_id = str(conversation_id)
        state = self.conversation_state.get(conversation_id, {})
        if state.get("stage") == "intro":
            state["stage"] = "title_prompt"
            self.conversation_state[conversation_id] = state
            save_conversation_state(self.conversation_state)
            return {
                "response": "Oh, sorry to hear that! Could you briefly tell me what the error is?"
            }

        if not state:
            return {"response": "Conversation not found. Please start a new conversation."}
        
        if state.get("stage") == "title_prompt":
            extracted_info = self._extract_ticket_info(user_input, target_field="title")
            if "title" in extracted_info:
                state["ticket_info"]["title"] = extracted_info["title"]
                state["stage"] = "fields_collected"
                state["current_field_index"] = 1
                self.conversation_state[conversation_id] = state
                save_conversation_state(self.conversation_state)
                return {
                    "response": "Got it. Can you elaborate a bit more on this issue? What exactly is happening, and when does it occur?"
                }
            else:
                return {"response": "Could you briefly describe the issue you're facing?"}

        if state.get("upload_done_asked", False):
            if user_input.strip().lower() in ["yes", "done"]:
                return self.finalize_ticket(conversation_id)
            elif user_input.strip().lower() in ["no"]:
                return {
                    "response": "Please upload an image, PDF, or link using the /upload-file endpoint."
                }
            else:
                return {
                    "response": "Please respond with 'yes' if you are done uploading or 'no' to upload more files."
                }

        if state.get("upload_asked", False):
            if user_input.strip().lower() in ["yes"]:
                if self.attachments.get(conversation_id, []):
                    state["upload_done_asked"] = True
                    self.conversation_state[conversation_id] = state
                    save_conversation_state(self.conversation_state)
                    return {
                        "response": "Are you done uploading files or links? (Respond with 'yes' or 'no')"
                    }
                return {
                    "response": "Please upload an image, PDF, or link using the /upload-file endpoint."
                }
            elif user_input.strip().lower() in ["no"]:
                state["stage"] = "ready_to_finalize"
                self.conversation_state[conversation_id] = state
                save_conversation_state(self.conversation_state)
                return {
                    "response": "Okay, I won’t ask for more uploads. Please use the /finalize endpoint to complete the ticket."
                }
            else:
                return {
                    "response": "Please respond with 'yes' to upload files or links or 'no' to proceed with ticket creation."
                }

        if state.get("fields_to_revise"):
            revision_field = state["fields_to_revise"][0]
            extracted_info = self._extract_ticket_info(user_input, target_field=revision_field)
            state["ticket_info"].update(extracted_info)
            state["fields_to_revise"].pop(0)
            if not state["fields_to_revise"]:
                state["current_field"] = None
                state["upload_asked"] = True
                state["stage"] = "fields_collected"
                self.conversation_state[conversation_id] = state
                save_conversation_state(self.conversation_state)
                return {
                    "response": "I've updated the field(s). Would you like to upload an image, PDF, or link related to the issue? (Respond with 'yes' or 'no')"
                }
            state["current_field"] = state["fields_to_revise"][0]
            self.conversation_state[conversation_id] = state
            save_conversation_state(self.conversation_state)
            return self._get_revision_prompt(state["current_field"], state.get("revision_guidance", []))

        ticket_info = state.get("ticket_info", {})
        current_field = state.get("field_order")[state.get("current_field_index")]
        extracted_info = self._extract_ticket_info(user_input, target_field=current_field)
        ticket_info.update(extracted_info)
        state["ticket_info"] = ticket_info
        logger.info(f"Extracted info for input '{user_input}': {extracted_info}")

        if current_field == "issue" and "issue" in ticket_info:
            issue_text = ticket_info["issue"].lower().strip()
            issue_words = issue_text.split()
    
            has_known_phrase = any(phrase in issue_text for phrase in self.issue_phrases)
            has_minimum_words = len(issue_words) >= 3
            has_action_verb = any(word in issue_text for word in ["fail", "not", "error", "crash", "unable", "stuck", "break", "freeze"])

            is_vague = not (has_known_phrase or has_minimum_words or has_action_verb)

            if is_vague:
                state["current_field"] = "issue"
                self.conversation_state[conversation_id] = state
                save_conversation_state(self.conversation_state)
                return {
                    "response": (
                        "The issue description is too vague. "
                        "Please provide more details, such as what isn't working, any error messages, or steps to reproduce the problem."
                    )
                }

        missing_fields = [f for f in state["field_order"] if f not in ticket_info or not ticket_info[f]]
        if not missing_fields:
            state["upload_asked"] = True
            state["stage"] = "fields_collected"
            state["current_field"] = None
            state["current_field_index"] = 0
            self.conversation_state[conversation_id] = state
            save_conversation_state(self.conversation_state)
            return {
                "response": "I've collected all the information. Would you like to upload an image, PDF, or link related to the issue? (Respond with 'yes' or 'no')"
            }

        next_field = missing_fields[0]
        state["current_field"] = next_field
        state["current_field_index"] = state["field_order"].index(next_field)
        self.conversation_state[conversation_id] = state
        save_conversation_state(self.conversation_state)

        prompts = {
            "title": "Please provide a brief title for the issue (e.g., 'Login Page Crash').",
            "issue": "Please describe the issue in detail (e.g., what isn't working, error messages, or steps to reproduce).",
            "priority": f"What priority would you assign? ({', '.join(self.valid_priorities).title()})",
            "component": f"Which component or system is affected? (Options: {', '.join(c.title() for c in self.valid_components)})",
            "issue_type": "Is this a 'Task' or a 'Bug'?"
        }

        if current_field == "priority" and user_input.strip().lower() not in self.valid_priorities:
            return {"response": f"Please specify a valid priority: {', '.join(self.valid_priorities).title()}"}

        if current_field == "component" and not any(c.lower() in user_input.lower() for c in self.valid_components):
            return {"response": f"Please specify a valid component: {', '.join(c.title() for c in self.valid_components)}"}

        if current_field == "issue_type" and user_input.strip().title() not in self.valid_issue_types:
            return {"response": f"Please specify a valid issue type: {', '.join(self.valid_issue_types)}"}

        return {"response": prompts[next_field]}

    def _extract_ticket_info(self, user_input: str, target_field: str = None) -> Dict:
        ticket_info = {}
        doc = self.nlp(user_input.lower())
        logger.info(f"Processing input for ticket info extraction: {user_input}, target_field: {target_field}")

        if target_field == "title":
            title = user_input.strip().title()
            if len(title) > 60:
                title = title[:57] + "..."
            if len(title) >= 5:
                ticket_info["title"] = title
            return ticket_info

        if target_field == "issue":
            ticket_info["issue"] = user_input.strip()
            ticket_info["issue_type"] = self._classify_issue_type(user_input)
            return ticket_info

        if target_field == "priority":
            for priority in self.valid_priorities:
                if priority in user_input.lower():
                    ticket_info["priority"] = priority
                    break
            return ticket_info

        if target_field == "component":
            for component in self.valid_components:
                if component.lower() in user_input.lower():
                    ticket_info["component"] = component.title()
                    break
            return ticket_info

        if target_field == "issue_type":
            issue_type = user_input.strip().title()
            if issue_type in self.valid_issue_types:
                ticket_info["issue_type"] = issue_type
            else:
                ticket_info["issue_type"] = self._classify_issue_type(user_input)
            return ticket_info

        ticket_info["issue"] = user_input.strip()
        ticket_info["issue_type"] = self._classify_issue_type(user_input)

        core_issue = None
        for sent in doc.sents:
            for phrase in self.issue_phrases:
                if phrase in sent.text:
                    core_issue = sent.text.strip()
                    break
            if core_issue:
                break

        if core_issue:
            core_issue = re.sub(
                r'(?:create a ticket for|open ticket|report|submit issue|file issue|' +
                r'|'.join(self.valid_priorities) + r'|' +
                r'|'.join(self.valid_components) + r')',
                '',
                core_issue,
                flags=re.IGNORECASE
            ).strip()
            core_issue = re.sub(r'\s+', ' ', core_issue).strip(',.')
            title_parts = []
            subject = None
            predicate = None
            for chunk in doc.noun_chunks:
                if chunk.text.lower() in core_issue.lower() and len(chunk.text) > 3:
                    subject = chunk.text.title()
                    title_parts.append(subject)
                    break
            for i, token in enumerate(doc):
                if token.pos_ in ["VERB", "ADJ"] and token.text.lower() in core_issue.lower():
                    if i > 0 and doc[i-1].text.lower() in ["not", "never"]:
                        predicate = f"{doc[i-1].text.title()} {token.text.title()}"
                    else:
                        predicate = token.text.title()
                    title_parts.append(predicate)
                    break
            title = " ".join(title_parts).strip() or core_issue.title()
            if title and len(title) >= 5:
                if len(title) > 60:
                    title = title[:57] + "..."
                ticket_info["title"] = title

        for priority in self.valid_priorities:
            if priority in user_input.lower():
                ticket_info["priority"] = priority
                break

        for component in self.valid_components:
            if component.lower() in user_input.lower():
                ticket_info["component"] = component.title()
                break

        logger.info(f"Extracted ticket info: {ticket_info}")
        return ticket_info

    def _classify_issue_type(self, text: str) -> str:
        text_lower = text.lower()
        bug_score = sum(1 for keyword in self.bug_keywords if keyword in text_lower)
        task_score = sum(1 for keyword in self.task_keywords if keyword in text_lower)

        doc = self.nlp(text_lower)
        for token in doc:
            if token.lemma_ in ["fix", "resolve", "correct"] and bug_score == 0:
                bug_score += 1
            if token.lemma_ in ["add", "implement", "create", "enhance"] and task_score == 0:
                task_score += 1

        logger.info(f"Issue classification - Bug score: {bug_score}, Task score: {task_score}, Text: {text}")

        if bug_score > task_score:
            return "Bug"
        elif task_score > bug_score:
            return "Task"
        else:
            return JIRA_ISSUE_TYPE if JIRA_ISSUE_TYPE in self.valid_issue_types else "Task"

    def _get_revision_prompt(self, field: str, guidance: List[str]) -> Dict:
        relevant_guidance = [g for g in guidance if field.lower() in g.lower()]
        prompts = {
            "title": "Please provide a revised title for the issue (e.g., 'Login Page Crash').",
            "issue": "Please provide a revised detailed description of the issue.",
            "priority": f"Please provide a revised priority ({', '.join(self.valid_priorities).title()}).",
            "component": f"Please provide a revised component ({', '.join(c.title() for c in self.valid_components)}).",
            "issue_type": f"Please specify if this is a 'Task' or 'Bug'."
        }
        prompt = relevant_guidance[0] if relevant_guidance else prompts.get(field, f"Please provide a revised {field}.")
        return {"response": prompt}

    def finalize_ticket(self, conversation_id: str) -> Dict:
        conversation_id = str(conversation_id)
        ticket_info = self.conversation_state.get(conversation_id, {}).get("ticket_info", {})
        attachments = self.attachments.pop(conversation_id, []) if conversation_id in self.attachments else []
        metadata = self.attachments_metadata.pop(conversation_id, []) if conversation_id in self.attachments_metadata else []

        if not ticket_info:
            return {"error": "No ticket information found for this conversation."}

        ticket_info["attachments"] = []
        for attachment, meta in zip(attachments, metadata):
            if meta.get("type") == "file":
                ticket_info["attachments"].append({
                    "filename": attachment,
                    "metadata": meta
                })
            elif meta.get("type") == "link":
                ticket_info["attachments"].append({
                    "url": attachment,
                    "metadata": meta
                })

        ticket = {
            "id": f"TICKET-{hash(conversation_id) % 10000:04}",
            "resolution": "",
            **{k: v for k, v in ticket_info.items() if k in self.required_fields or k == "issue_type"},
            "attachments": ticket_info["attachments"]
        }

        analysis = self.ticket_analyzer.execute("analyze_ticket", ticket)
        if analysis["status"] == "incomplete":
            guidance = self.ticket_analyzer.execute("generate_guidance", analysis)
            fields_to_revise = []
            for issue in analysis["quality_issues"]:
                for field in self.required_fields + ["issue_type"]:
                    if field.capitalize() in issue:
                        fields_to_revise.append(field)
                        break
            if fields_to_revise:
                self.conversation_state[conversation_id] = {
                    "stage": "revision_needed",
                    "fields_to_revise": fields_to_revise,
                    "current_field": fields_to_revise[0],
                    "revision_guidance": guidance["guidance"],
                    "ticket_info": ticket_info,
                    "field_order": ["title", "issue", "priority", "component", "issue_type"],
                    "current_field_index": 0
                }
                save_conversation_state(self.conversation_state)
                return {
                    "message": f"Please revise the following field: {fields_to_revise[0]}.",
                    "analysis": analysis,
                    "guidance": self._get_revision_prompt(fields_to_revise[0], guidance["guidance"])
                }
            return {
                "message": "Ticket analysis found issues. Please address them and finalize again.",
                "analysis": analysis,
                "guidance": guidance
            }

        self.finalized_tickets[conversation_id] = ticket
        self.conversation_state[conversation_id] = {"stage": "finalized"}
        save_conversation_state(self.conversation_state)

        ticket_details = (
            f"✓ Ticket {ticket['id']} created successfully.\n"
            f"Title: {ticket.get('title', 'N/A')}\n"
            f"Issue: {ticket.get('issue', 'N/A')}\n"
            f"Priority: {ticket.get('priority', 'N/A').title()}\n"
            f"Component: {ticket.get('component', 'N/A')}\n"
            f"Issue Type: {ticket.get('issue_type', JIRA_ISSUE_TYPE)}"
        )

        return {
            "message": "Thanks! I’ve created the ticket.",
            "ticket": ticket,
            "response": ticket_details
        }

    def save_attachment(self, file: Optional[UploadFile], conversation_id: str, link: Optional[str] = None) -> str:
        conversation_id = str(conversation_id)
        try:
            if file:
                file_ext = os.path.splitext(file.filename)[1].lower()
                if file_ext not in ALLOWED_EXTENSIONS:
                    raise ValueError(f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} are allowed.")

                file_size = 0
                temp_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename.replace('/', '_')}")
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                    file_size = os.path.getsize(temp_path)

                mime_type, _ = mimetypes.guess_type(file.filename)
                metadata = {
                    "filename": file.filename,
                    "size_bytes": file_size,
                    "mime_type": mime_type or "application/octet-stream",
                    "type": "file"
                }

                filename = f"{conversation_id}{file.filename.replace('/', '')}"
                filepath = os.path.join(UPLOAD_DIR, filename)
                os.rename(temp_path, filepath)
                self.attachments.setdefault(conversation_id, []).append(filename)
                self.attachments_metadata.setdefault(conversation_id, []).append(metadata)
                logger.info(f"File saved: {filename}, Metadata: {metadata}, conversation_id: {conversation_id}")
                return f"File '{file.filename}' uploaded successfully."
            elif link and validators.url(link):
                metadata = {
                    "url": link,
                    "type": "link"
                }
                self.attachments.setdefault(conversation_id, []).append(link)
                self.attachments_metadata.setdefault(conversation_id, []).append(metadata)
                logger.info(f"Link saved: {link}, Metadata: {metadata}, conversation_id: {conversation_id}")
                return f"Link '{link}' saved successfully."
            else:
                raise ValueError("Invalid link or no file provided.")
        except (OSError, PermissionError) as e:
            logger.error(f"File save error: {str(e)}")
            raise ValueError(f"Failed to save attachment: {str(e)}")

    def get_conversation_status(self, conversation_id: str) -> Dict:
        conversation_id = str(conversation_id)
        if conversation_id in self.finalized_tickets:
            return {
                "status": "finalized",
                "ticket": self.finalized_tickets[conversation_id]
            }
        elif conversation_id in self.conversation_state:
            return {
                "status": "active",
                "state": self.conversation_state[conversation_id],
                "attachments": self.attachments.get(conversation_id, []),
                "attachments_metadata": self.attachments_metadata.get(conversation_id, [])
            }
        else:
            return {
                "status": "not_found",
                "message": "No active or completed conversation found."
            }

# ---- Orchestrator Agent ----
class OrchestratorAgent:
    def _init_(self, rag_chatbot, ticket_creation_agent):
        self.rag_chatbot = rag_chatbot
        self.ticket_creation_agent = ticket_creation_agent
        self.conversation_states = load_conversation_state()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            logger.error(f"Failed to load spaCy model 'en_core_web_sm': {str(e)}")
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")
        self.ticket_keywords = [
            "create ticket", "new ticket", "submit issue",
            "report bug", "open ticket", "file issue"
        ]
        self.issue_keywords = [
            "not working", "not loading", "incorrect results",
            "error", "bug", "problem with", "issue with"
        ]
        self.priority_keywords = ["low", "medium", "high", "critical"]
        self.component_keywords = ["authentication", "dashboard", "payment gateway", "user profile", "notifications", "search"]

    def _is_ticket_creation_request(self, user_input: str, conversation_id: str) -> bool:
        user_input_lower = user_input.lower()
        keyword_match = any(keyword in user_input_lower for keyword in self.ticket_keywords)
        issue_match = any(keyword in user_input_lower for keyword in self.issue_keywords)
        priority_match = any(keyword in user_input_lower for keyword in self.priority_keywords)
        component_match = any(keyword in user_input_lower for keyword in self.component_keywords)

        doc = self.nlp(user_input)
        intent_score = 0.0

        for token in doc:
            if token.dep_ == "ROOT" and token.lemma_ in ["create", "report", "submit", "file", "open"]:
                intent_score += 0.5
            if token.text.lower() in ["ticket", "issue", "bug", "problem"]:
                intent_score += 0.3
            if token.ent_type_ in ["PRODUCT", "EVENT", "ORG"]:
                intent_score += 0.2
            if token.text.lower() in self.issue_keywords:
                intent_score += 0.2
            if token.text.lower() in self.priority_keywords:
                intent_score += 0.2
            if token.text.lower() in self.component_keywords:
                intent_score += 0.1

        is_ongoing_ticket = conversation_id in self.ticket_creation_agent.conversation_state

        logger.info(f"Intent detection for '{user_input}': keyword_match={keyword_match}, issue_match={issue_match}, priority_match={priority_match}, component_match={component_match}, intent_score={intent_score}, is_ongoing_ticket={is_ongoing_ticket}")
        
        return keyword_match or issue_match or priority_match or (component_match and is_ongoing_ticket) or intent_score > 0.6

    def process_request(self, user_input: str, conversation_id: Optional[str] = None) -> Dict:
        if not user_input.strip():
            return {"error": "User input cannot be empty"}

        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        conversation_id = str(conversation_id)

        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = {
                "mode": "general",
                "ticket_info": {},
                "last_response": None,
                "context_history": []
            }

        state = self.conversation_states[conversation_id]
        state.setdefault("context_history", [])
        state["context_history"].append(user_input)
        state["context_history"] = state["context_history"][-5:]

        if "mode" not in state:
            logger.warning(f"Mode missing for conversation {conversation_id}, setting to general")
            state["mode"] = "general"

        if state["mode"] == "ticket_creation":
            response = self.ticket_creation_agent.create_ticket(user_input, conversation_id)
            if "response" in response and "Ticket" in response.get("response", ""):
                state["mode"] = "general"
                state["ticket_info"] = {}
            state["last_response"] = response
            self.conversation_states[conversation_id] = state
            save_conversation_state(self.conversation_states)
            return response

        if self._is_ticket_creation_request(user_input, conversation_id):
            state["mode"] = "ticket_creation"
            response = self.ticket_creation_agent.create_ticket(user_input, conversation_id)
            state["last_response"] = response
            self.conversation_states[conversation_id] = state
            save_conversation_state(self.conversation_states)
            return response

        if state.get("mode") != "ticket_creation":
            state["mode"] = "ticket_creation"
            self.conversation_states[conversation_id] = state
            save_conversation_state(self.conversation_states)
            response = self.ticket_creation_agent.create_ticket(user_input, conversation_id)
            return response

    def reset_conversation(self, conversation_id: str) -> Dict:
        conversation_id = str(conversation_id)
        if conversation_id in self.conversation_states:
            self.conversation_states[conversation_id] = {
                "mode": "general",
                "ticket_info": {},
                "last_response": None,
                "context_history": []
            }
            if conversation_id in self.ticket_creation_agent.conversation_state:
                self.ticket_creation_agent.conversation_state.pop(conversation_id, None)
                self.ticket_creation_agent.attachments.pop(conversation_id, None)
                self.ticket_creation_agent.attachments_metadata.pop(conversation_id, None)
            logger.info(f"Conversation reset: {conversation_id}")
            save_conversation_state(self.conversation_states)
            return {"message": f"Conversation {conversation_id} has been reset to general mode."}
        return {"message": f"No active conversation found for {conversation_id}."}

# Initialize agents
model = SentenceTransformer('all-MiniLM-L6-v2')
ticket_analyzer = TicketAnalysisAgent()
ticket_agent = TicketCreationAgent(
    ticket_creation_api_url="https://api.example.com/create",
    ticket_analyzer=ticket_analyzer
)
rag_chatbot = MockRAGChatbot()
orchestrator = OrchestratorAgent(rag_chatbot, ticket_agent)

# ---- Utility Functions ----
def extract_tickets(data, results=None) -> List[Dict]:
    if results is None:
        results = []
    if isinstance(data, dict):
        if "title" in data and "issue" in data:
            text = f"{data['title']} {data['issue']}"
            results.append({
                "text": text,
                "id": data.get("id"),
                "title": data.get("title"),
                "issue": data.get("issue"),
                "priority": data.get("priority"),
                "assignee": data.get("assignee", {}),
                "resolution": data.get("resolution", ""),
                "attachments": data.get("attachments", []),
                "issue_type": data.get("issue_type", JIRA_ISSUE_TYPE)
            })
        for value in data.values():
            extract_tickets(value, results)
    elif isinstance(data, list):
        for item in data:
            extract_tickets(item, results)
    return results

def save_index_and_meta(embeddings: np.ndarray, tickets: List[Dict]):
    try:
        if not os.access(os.path.dirname(INDEX_FILE) or ".", os.W_OK):
            raise PermissionError(f"No write permission for {INDEX_FILE}")
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(tickets, f)
        logger.info(f"Saved FAISS index and metadata for {len(tickets)} tickets")
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to save index and metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save index: {str(e)}")

def load_index_and_meta() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        logger.info("Loaded FAISS index and metadata")
        return index, metadata
    except FileNotFoundError:
        logger.error("Index or metadata file not found")
        raise HTTPException(status_code=500, detail="Index or metadata not found. Please index tickets first.")

# ---- Jira Integration Function ----
def get_assignable_users_for_component(component: str) -> Optional[str]:
    """Fetch assignable users for the project and select one based on component."""
    try:
        # Fetch assignable users for the project
        users = jira_client.search_assignable_users_for_projects(
            username="",  # Empty to get all users
            projectKeys=JIRA_PROJECT_KEY,
            maxResults=100
        )
        if not users:
            logger.warning(f"No assignable users found for project {JIRA_PROJECT_KEY}")
            return DEFAULT_ASSIGNEE

        # Get roles associated with the component
        target_roles = COMPONENT_ROLE_MAPPING.get(component.lower(), [])
        if not target_roles:
            logger.warning(f"No roles mapped for component {component}")
            return DEFAULT_ASSIGNEE

        # Filter users based on roles (simulated, as Jira API doesn't directly provide roles)
        # In a real scenario, you may need to maintain a user-role mapping or query user groups
        suitable_users = []
        for user in users:
            # Simulate role check (replace with actual role/group fetching if available)
            # Here, we assume user.displayName or user.emailAddress contains role hints for simplicity
            user_roles = []  # Placeholder: Fetch actual roles from Jira groups or external mapping
            if any(role in user.displayName or role in user.emailAddress for role in target_roles):
                suitable_users.append(user.accountId)

        if suitable_users:
            # Select a random suitable user to distribute workload
            import random
            selected_user = random.choice(suitable_users)
            logger.info(f"Selected user {selected_user} for component {component}")
            return selected_user
        else:
            logger.warning(f"No users with roles {target_roles} found for component {component}")
            return DEFAULT_ASSIGNEE

    except Exception as e:
        logger.error(f"Failed to fetch assignable users: {str(e)}")
        return DEFAULT_ASSIGNEE

def create_jira_ticket(ticket: Dict) -> Dict:
    try:
        description = ticket.get("issue", "")
        attachments = ticket.get("attachments", [])
        links = []
        for att in attachments:
            if att.get("metadata", {}).get("type") == "link":
                url = att.get("url") or att.get("metadata", {}).get("url")
                if url:
                    links.append(url)

        if links:
            description += "\n\nRelated Links:\n" + "\n".join(f"- {link}" for link in links)

        issue_type = ticket.get("issue_type", JIRA_ISSUE_TYPE if JIRA_ISSUE_TYPE in ["Task", "Bug"] else "Task")

        issue_dict = {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": ticket.get("title", "New Ticket"),
            "description": description,
            "issuetype": {"name": issue_type},
            "priority": {"name": ticket.get("priority", "Medium").capitalize()}
        }
        if ticket.get("component"):
            issue_dict["components"] = [{"name": ticket["component"]}]
            # Assign ticket to a user based on component
            assignee_id = get_assignable_users_for_component(ticket["component"])
            if assignee_id:
                issue_dict["assignee"] = {"accountId": assignee_id}
            else:
                logger.warning(f"No assignee found for component {ticket['component']}, using default or unassigned")

        new_issue = jira_client.create_issue(fields=issue_dict)
        logger.info(f"Created Jira ticket: {new_issue.key} as {issue_type}")

        for attachment in attachments:
            if attachment["metadata"].get("type") == "file":
                filename = attachment["filename"]
                filepath = os.path.join(UPLOAD_DIR, filename)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "rb") as f:
                            jira_client.add_attachment(issue=new_issue, attachment=f, filename=attachment["metadata"]["filename"])
                        logger.info(f"Attached {filename} to Jira ticket {new_issue.key}")
                    except Exception as e:
                        logger.error(f"Failed to attach {filename} to Jira ticket {new_issue.key}: {str(e)}")
                else:
                    logger.warning(f"Attachment file not found: {filepath}")

        return {
            "jira_key": new_issue.key,
            "jira_url": f"{JIRA_URL}/browse/{new_issue.key}",
            "issue_type": issue_type,
            "assignee": issue_dict.get("assignee", {}).get("accountId", "Unassigned"),
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Failed to create Jira ticket: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create Jira ticket: {str(e)}")

# ---- API Endpoints ----
@app.post("/start-chat")
async def start_chat(
    conversation_id: str = Form(...)
):
    try:
        response = ticket_agent.start_conversation(conversation_id)
        return {"message": response, "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Error in start_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")

@app.post("/chat")
async def chat(
    conversation_id: str = Form(...),
    user_input: str = Form(...)
):
    try:
        if not user_input.strip():
            return JSONResponse(status_code=400, content={"error": "User input cannot be empty."})
        if conversation_id not in ticket_agent.conversation_state:
            ticket_agent.start_conversation(conversation_id)
        response = orchestrator.process_request(user_input, conversation_id=conversation_id)
        return response
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat input: {str(e)}")

@app.post("/upload-file")
async def upload_file(
    conversation_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    link: Optional[str] = Form(None)
):
    try:
        if conversation_id not in ticket_agent.conversation_state:
            return JSONResponse(status_code=400, content={"error": "Invalid conversation_id"})
        message = ticket_agent.save_attachment(file, conversation_id, link)
        return {"message": message}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/finalize")
async def finalize(conversation_id: str = Form(...)):
    try:
        conversation_id = str(conversation_id)
        if conversation_id in ticket_agent.finalized_tickets:
            return {
                "message": "This conversation has already been finalized.",
                "ticket": ticket_agent.finalized_tickets[conversation_id]
            }
        if conversation_id not in ticket_agent.conversation_state:
            return JSONResponse(status_code=400, content={"error": "Conversation not found"})
        response = ticket_agent.finalize_ticket(conversation_id)
        if "ticket" in response:
            orchestrator.conversation_states[conversation_id]["mode"] = "general"
            orchestrator.conversation_states[conversation_id]["ticket_info"] = {}
            save_conversation_state(orchestrator.conversation_states)
        return response
    except Exception as e:
        logger.error(f"Error in finalize: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to finalize ticket: {str(e)}")

@app.post("/api/ticket/validate")
async def validate_ticket(conversation_id: str = Form(...)):
    logger.info(f"Validating ticket for conversation {conversation_id}")
    ticket = ticket_agent.finalized_tickets.get(conversation_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="No finalized ticket found for this conversation.")

    analysis = ticket_analyzer.execute("analyze_ticket", ticket)
    guidance = ticket_analyzer.execute("generate_guidance", analysis)

    if conversation_id in ticket_agent.conversation_state:
        state = ticket_agent.conversation_state[conversation_id]
        if state["stage"] != "finalized":
            return {"error": f"Invalid stage: {state['stage']}. Expected 'finalized'. Complete /finalize first."}
        if analysis["status"] == "success":
            state["stage"] = "validated"
            ticket_agent.conversation_state[conversation_id] = state
            save_conversation_state(ticket_agent.conversation_state)

    return {
        "message": "Analysis successful",
        "analysis": analysis,
        "guidance": guidance
    }

@app.post("/api/similarity/search")
async def search_similar_ticket(
    query: str = Form(...),
    conversation_id: str = Form(...)
):
    logger.info(f"Received similarity search for conversation: {conversation_id}, Query: {query}")
    if conversation_id not in ticket_agent.conversation_state:
        raise HTTPException(
            status_code=404,
            detail={"error": "Conversation not found", "message": f"No conversation found with ID: {conversation_id}."}
        )

    state = ticket_agent.conversation_state[conversation_id]
    if state.get("stage") == "started":
        finalize_response = ticket_agent.finalize_ticket(conversation_id)
        if "ticket" not in finalize_response:
            raise HTTPException(status_code=400, detail="Auto-finalize failed. Please fix ticket info.")
        ticket = finalize_response["ticket"]
        ticket_agent.finalized_tickets[conversation_id] = ticket
        state["stage"] = "finalized"
        ticket_agent.conversation_state[conversation_id] = state
        save_conversation_state(ticket_agent.conversation_state)

    if state["stage"] not in ["finalized", "validated"]:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {state['stage']}. Expected 'finalized' or 'validated'.")

    session_ticket = ticket_agent.finalized_tickets.get(conversation_id)
    if not session_ticket:
        raise HTTPException(status_code=404, detail="No finalized ticket found in session")

    session_ticket["priority"] = session_ticket.get("priority", "").lower()
    search_query = query.strip() if query.strip() else session_ticket["issue"]

    try:
        index, ticket_data = load_index_and_meta()
    except HTTPException:
        analysis = ticket_analyzer.execute("analyze_ticket", session_ticket)
        if analysis["status"] != "success":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Cannot index invalid ticket.",
                    "analysis": analysis,
                    "guidance": ticket_analyzer.execute("generate_guidance", analysis)
                }
            )
        state["stage"] = "searched"
        ticket_agent.conversation_state[conversation_id] = state
        save_conversation_state(ticket_agent.conversation_state)
        jira_response = create_jira_ticket(session_ticket)
        return {
            "message": "No index available. Ticket created in Jira backlog.",
            "created_ticket": session_ticket,
            "jira": jira_response,
            "next_step": f"POST /api/similarity/index with conversation_id: {conversation_id}",
            "status": "pending_index"
        }

    try:
        query_embedding = model.encode([search_query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    D, I = index.search(query_embedding, k=3)
    results = []
    for score, idx in zip(D[0], I[0]):
        if score >= 0.75:
            ticket = ticket_data[idx]
            results.append({
                "id": ticket.get("id", ""),
                "title": ticket.get("title", ""),
                "issue": ticket.get("issue", ""),
                "priority": ticket.get("priority", ""),
                "assignee": ticket.get("assignee", {}),
                "resolution": ticket.get("resolution", ""),
                "issue_type": ticket.get("issue_type", ""),
                "similarity_score": round(float(score), 4)
            })

    state["stage"] = "searched"
    ticket_agent.conversation_state[conversation_id] = state
    save_conversation_state(ticket_agent.conversation_state)

    if results:
        summarized_resolution = ticket_analyzer.summarize_resolutions(results)
        return {
            "message": "Similar tickets found.",
            "matches": results,
            "summarized_resolution": summarized_resolution,
            "status": "success"
        }

    jira_response = create_jira_ticket(session_ticket)
    return {
        "message": "No similar tickets found. Ticket created in Jira backlog.",
        "created_ticket": session_ticket,
        "jira": jira_response,
        "next_step": f"POST /api/similarity/index with conversation_id: {conversation_id}",
        "status": "completed"
    }

@app.post("/api/similarity/index")
async def index_new_tickets(conversation_id: str = Form(...)):
    logger.info(f"Indexing ticket for conversation {conversation_id}")
    session_ticket = ticket_agent.finalized_tickets.get(conversation_id)
    if not session_ticket:
        raise HTTPException(status_code=404, detail="No ticket information found in session")

    session_ticket["priority"] = session_ticket.get("priority", "").lower()
    analysis = ticket_analyzer.execute("analyze_ticket", session_ticket)
    if analysis["status"] != "success":
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Cannot index ticket.",
                "analysis": analysis,
                "guidance": ticket_analyzer.execute("generate_guidance", analysis)
            }
        )

    ticket_for_indexing = {
        "text": f"{session_ticket['title']} {session_ticket['issue']}",
        "id": session_ticket["id"],
        "title": session_ticket["title"],
        "issue": session_ticket["issue"],
        "priority": session_ticket["priority"],
        "assignee": session_ticket.get("assignee", {}),
        "resolution": session_ticket.get("resolution", ""),
        "attachments": session_ticket.get("attachments", []),
        "issue_type": session_ticket.get("issue_type", JIRA_ISSUE_TYPE)
    }

    try:
        index, ticket_data = load_index_and_meta()
        ticket_data.append(ticket_for_indexing)
    except HTTPException:
        ticket_data = [ticket_for_indexing]

    try:
        embeddings = np.array([model.encode([ticket["text"]]) for ticket in ticket_data])
        save_index_and_meta(embeddings, ticket_data)
    except Exception as e:
        logger.error(f"Failed to encode tickets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to index tickets: {str(e)}")

    if conversation_id in ticket_agent.conversation_state:
        state = ticket_agent.conversation_state[conversation_id]
        state["stage"] = "created"
        ticket_agent.conversation_state[conversation_id] = state
        save_conversation_state(ticket_agent.conversation_state)

    return {
        "message": f"Successfully indexed ticket {session_ticket['id']}.",
        "indexed_ticket": {
            "id": session_ticket["id"],
            "title": session_ticket["title"],
            "issue": session_ticket["issue"],
            "priority": session_ticket["priority"],
            "issue_type": session_ticket["issue_type"],
            "ticket": session_ticket,
            "status": "indexed",
            "ticket_id": session_ticket["id"]
        }
    }

@app.get("/api/ticket/{ticket_id}/status")
async def get_ticket_status(ticket_id: str):
    try:
        ticket_status = ticket_agent.get_conversation_status(ticket_id)
        orchestrator_status = orchestrator.conversation_states.get(ticket_id, {"status": "not_found"})
        if ticket_status["status"] == "not_found" and orchestrator_status["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Ticket not found")
        return {
            "ticket_status": ticket_status,
            "orchestrator_status": orchestrator_status
        }
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/reset_conversation")
async def reset_conversation(conversation_id: str = Form(...)):
    try:
        response = orchestrator.reset_conversation(conversation_id)
        return response
    except Exception as e:
        logger.error(f"Error in reset_conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset conversation: {str(e)}")

@app.post("/api/conversations")
async def list_conversations():
    try:
        conversations = [
            {
                "conversation_id": cid,
                "state": state,
                "ticket": ticket_agent.finalized_tickets.get(cid),
                "status": "finalized" if cid in ticket_agent.finalized_tickets else "active"
            }
            for cid, state in ticket_agent.conversation_state.items()
        ]
        logger.info(f"Retrieved {len(conversations)} active conversations")
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error in list_conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error at {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)}
    )