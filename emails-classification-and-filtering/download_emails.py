from __future__ import print_function
import csv, json, re, os
import base64
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

# Scope to read Gmail
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def authentication():
    """
    Authenticates user.

    Returns:
        service resource
    """
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server()
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build(
        "gmail",
        "v1",
        credentials=creds,
    )
    return service


def get_email_body(payload):
    """
    Extract body from the email payload. Check if string contains html or not.
    If it cotains html parse html and return text only otherwise strip whitespace characters
    """
    body = ""

    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data")
                if data:
                    body += base64.urlsafe_b64decode(data).decode(
                        "utf-8", errors="ignore"
                    )
            elif part["mimeType"] == "text/html":
                data = part["body"].get("data")
                if data:
                    html = base64.urlsafe_b64decode(data).decode(
                        "utf-8", errors="ignore"
                    )
                    body += BeautifulSoup(html, "html.parser").get_text()
            elif "parts" in part:  # Recursive for nested parts
                body += get_email_body(part)
    else:
        data = payload["body"].get("data")
        if data:
            body += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    soup = BeautifulSoup(body, "html.parser")
    text = soup.get_text(strip=True).replace("\t", "").replace("\r\n", "")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\r+", " ", text)
    text = text.encode("ascii", errors="ignore").strip().decode("ascii")
    return text


def fetch_email(service, start_date: str, end_date: str):
    """
    Fetch email and download them as a json file

    Args:
        service: To fetch all emails in the below specified date range.
        start_date: In `YYYY/MM/DD` format
        end_date: In `YYYY/MM/DD` format
    """

    query = f"after:{start_date} before:{end_date}"

    all_messages = []
    page_token = None

    while True:
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query, pageToken=page_token)
            .execute()
        )

        messages = results.get("messages", [])
        all_messages.extend(messages)
        page_token = results.get("nextPageToken")

        if not page_token:
            break

    print(f"Found {len(all_messages)} emails from {start_date} to now.")

    for i, msg in enumerate(all_messages, 1):
        msg_data = (
            service.users()
            .messages()
            .get(userId="me", id=msg["id"], format="full")
            .execute()
        )

        headers = msg_data["payload"]["headers"]

        subject = next(
            (h["value"] for h in headers if h["name"] == "Subject"), "(No Subject)"
        )

        sender = next(
            (h["value"] for h in headers if h["name"] == "From"), "(Unknown Sender)"
        )
        date = next((h["value"] for h in headers if h["name"] == "Date"), "")
        body = get_email_body(msg_data["payload"])
        if os.path.exists("./emails") is False:
            os.mkdir("./emails")
        with open(f"./emails/{i}.json", "w+") as f:
            json.dump(
                {"subject": subject, "sender": sender, "date": date, "body": body}, f
            )
        if i % 50 == 0:
            print(f"Downloaded and saved {i} emails in json format")


if __name__ == "__main__":
    service = authentication()
    fetch_email(service, "2025/01/01", "2025/08/16")
