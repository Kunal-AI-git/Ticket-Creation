## Ticket Creation
A FastAPI-based application for creating, managing, and indexing tickets with Jira integration and similarity search capabilities using FAISS and Sentence Transformers.
Table of Contents

## Overview
This project provides a RESTful API for ticket creation and management, integrated with Jira for ticket submission and FAISS for similarity search. It includes natural language processing with spaCy for ticket analysis and intent detection, and supports file uploads (images, PDFs) and links as ticket attachments.

## Features

Ticket Creation: Interactive ticket creation with guided prompts for title, issue, priority, component, and issue type.
Jira Integration: Automatically creates tickets in Jira with assignee logic based on components.
Similarity Search: Uses FAISS and Sentence Transformers to find similar tickets based on embeddings.
File Uploads: Supports uploading images, PDFs, and links as ticket attachments.
Conversation Management: Maintains conversation state for multi-turn ticket creation.
Ticket Validation: Analyzes tickets for completeness and quality before submission.

## Installation

Install dependencies:
pip install -r requirements.txt


Install the spaCy language model:
python -m spacy download en_core_web_sm


Set up environment variables (see Configuration).

Run the application:
uvicorn main:app --reload



## Configuration
Create a .env file in the project root with the following variables:
JIRA_DOMAIN=your-jira-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-jira-api-token
JIRA_PROJECT_KEY=your-project-key
JIRA_ISSUE_TYPE=Task
JIRA_DEFAULT_ASSIGNEE=your-default-assignee-account-id


JIRA_DOMAIN: Your Jira instance domain (e.g., yourcompany.atlassian.net).
JIRA_EMAIL: Your Jira account email.
JIRA_API_TOKEN: API token from Jira (generate at https://id.atlassian.com/manage-profile/security/api-tokens).
JIRA_PROJECT_KEY: The key of the Jira project to create tickets in.
JIRA_ISSUE_TYPE: Default issue type (Task or Bug).
JIRA_DEFAULT_ASSIGNEE: Default assignee account ID if no suitable assignee is found.

## Usage

Start the FastAPI server:
uvicorn main:app --host 0.0.0.0 --port 8000


Access the interactive API documentation at http://localhost:8000/docs.

Use the API endpoints to start a conversation, upload files, finalize tickets, search for similar tickets, and index tickets.


## API Endpoints

POST /start-chat: Start a new conversation for ticket creation.
POST /chat: Continue a conversation with user input to build a ticket.
POST /upload-file: Upload files (PNG, JPG, JPEG, PDF) or links for a ticket.
POST /finalize: Finalize a ticket after collecting all required information.
POST /api/ticket/validate: Validate a finalized ticket for completeness and quality.
POST /api/similarity/search: Search for similar tickets using FAISS.
POST /api/similarity/index: Index a finalized ticket for future similarity searches.
GET /api/ticket/{ticket_id}/status: Get the status of a ticket or conversation.
POST /reset_conversation: Reset a conversation to start over.
POST /api/conversations: List all active and finalized conversations.

## Directory Structure
ticket-management-api/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── .env                 # Environment variables (not tracked)
├── uploads/             # Directory for uploaded files
├── tickets_index.faiss  # FAISS index file
├── ticket_metadata.pkl  # Ticket metadata file
├── conversation_state.pkl # Conversation state file

## Dependencies
Listed in requirements.txt. Key dependencies include:

fastapi: For building the RESTful API.
uvicorn: ASGI server for running the FastAPI app.
pydantic: For data validation and modeling.
sentence-transformers: For generating text embeddings.
faiss-cpu: For efficient similarity search.
spacy: For natural language processing.
jira: For Jira API integration.
python-dotenv: For loading environment variables.

## License
This project is licensed under the MIT License.
