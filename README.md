# Study Tools API

A FastAPI-based service that integrates Anki and Notion study tools, with SQLite database for linking content between them. The service can be exposed via ngrok for ChatGPT plugin integration.

## Features

- Anki integration via AnkiConnect
- Notion API integration
- SQLite database for linking Anki cards with Notion pages
- ngrok tunneling for external access
- OpenAPI specification for ChatGPT plugin integration

## Prerequisites

- Python 3.8+
- Anki with AnkiConnect add-on installed
- Notion account with integration token
- [Optional] ngrok account for external access

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd study-tools-api
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file:
```bash
cp .env.example .env
```

5. Edit `.env` file with your configuration:
- Add your Notion API token
- Configure AnkiConnect URL if different
- Add ngrok auth token if using external access

## Running the Service

### Local Development

```bash
python main.py
```

This will start the service on `http://localhost:8000` with the following endpoints:
- Anki service: `http://localhost:8001`
- Notion service: `http://localhost:8002`
- Glue service: `http://localhost:8003`

### With ngrok (for ChatGPT Plugin)

1. Add your ngrok auth token to `.env`
2. Run the service:
```bash
python main.py
```

The service will automatically create an ngrok tunnel and display the public URL.

## API Documentation

Once the service is running, you can access:
- OpenAPI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## ChatGPT Plugin Integration

1. Get your ngrok URL from the console output
2. Use this URL as your plugin's API endpoint
3. The OpenAPI specification will be available at `{ngrok_url}/openapi.json`

## Windows-Specific Notes

1. Make sure Anki is running before starting the service
2. The default AnkiConnect URL (`http://localhost:8765`) should work
3. If using ngrok, ensure your Windows firewall allows the connection

## Troubleshooting

1. **AnkiConnect not responding**
   - Ensure Anki is running
   - Check if AnkiConnect add-on is installed
   - Verify the AnkiConnect URL in `.env`

2. **Notion API errors**
   - Verify your Notion integration token
   - Check if the integration has proper permissions

3. **Database errors**
   - Ensure write permissions in the project directory
   - Check if SQLite is properly installed

4. **ngrok connection issues**
   - Verify your ngrok auth token
   - Check Windows firewall settings
   - Ensure port 8000 is not blocked

## Docker Setup

To run the application using Docker:

1. Create a `.env` file in the root directory with the following variables:
   ```
   # Notion API Token (required)
   NOTION_API_TOKEN=your_notion_api_token_here

   # Ngrok Configuration (optional)
   NGROK_AUTHTOKEN=your_ngrok_auth_token_here
   NGROK_DOMAIN=your_custom_domain_here

   # Database Configuration
   DB_URL=sqlite:///./mapping.db
   ```

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

The API will be available at `http://localhost:8000`.

Note: Make sure you have Docker and Docker Compose installed on your system 