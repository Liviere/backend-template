"""
Email Tools for LangChain Agents

Provides agent tools for reading and sending emails via configured
IMAP/SMTP hosts. Integrates with EmailService and ImapService.

Tools:
- ReadUnseenEmailsTool: Fetch unread emails from mailbox
- SearchEmailsTool: Search emails by criteria
- SendEmailTool: Compose and send emails
- SummarizeEmailTool: Summarize email content using LLM

"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from inference_core.services.email_service import EmailService
    from inference_core.services.imap_service import ImapService

logger = logging.getLogger(__name__)


class EmailType(str, Enum):
    """Email classification types for summarization."""

    SPAM = "spam"
    NEWSLETTER = "newsletter"
    IMPORTANT = "important"
    TRANSACTIONAL = "transactional"
    PERSONAL = "personal"


EMAIL_TYPES = [et.value for et in EmailType]


def format_email_types_for_description() -> str:
    """Generate formatted list of email types for tool documentation."""
    return "Available email types: " + ", ".join(EMAIL_TYPES)


class EmailSummary(BaseModel):
    """Structured email summary response."""

    summary: str = Field(description="Concise summary of the email content")
    email_type: EmailType = Field(description="Classification of the email")
    requires_action: bool = Field(
        default=False, description="Whether the email requires user action/response"
    )
    key_points: List[str] = Field(
        default_factory=list, description="Key points extracted from the email"
    )


class ReadUnseenEmailsTool(BaseTool):
    """Tool for reading unread emails from configured mailboxes.

    Fetches unseen emails from IMAP server and returns structured data.
    Respects allowed_accounts configuration for agent access control.
    """

    name: str = "read_unseen_emails"
    description: str = """Fetch unread emails from a mailbox.
Use when you need to check for new emails or review inbox.

Arguments:
  - account_name (optional): Email account alias to use. If omitted, uses default account.
  - folder (optional): Mailbox folder to read from. Default: INBOX
  - limit (optional): Maximum number of emails to fetch. Default: 10
  - mark_as_read (optional): Whether to mark emails as read. Default: false

Returns structured email data including: UID, subject, sender, date, body preview, attachments info.
"""

    imap_service: Any = Field(exclude=True)
    allowed_accounts: Optional[List[str]] = Field(default=None, exclude=True)
    default_account: Optional[str] = Field(default=None, exclude=True)

    def _run(
        self,
        account_name: Optional[str] = None,
        folder: str = "INBOX",
        limit: int = 10,
        mark_as_read: bool = False,
    ) -> str:
        try:
            account = self._resolve_account(account_name)
            messages = self.imap_service.fetch_unseen_emails(
                host_alias=account,
                folder=folder,
                limit=limit,
                mark_as_read=mark_as_read,
            )
            return self._format_messages(messages, account, folder)

        except ValueError as e:
            logger.error("Account access error: %s", e)
            return f"✗ {e}"
        except Exception as e:
            logger.error("Failed to read emails: %s", e, exc_info=True)
            return f"✗ Failed to read emails: {e}"

    async def _arun(
        self,
        account_name: Optional[str] = None,
        folder: str = "INBOX",
        limit: int = 10,
        mark_as_read: bool = False,
    ) -> str:
        # ImapService is synchronous, wrap in thread executor
        return self._run(account_name, folder, limit, mark_as_read)

    def _resolve_account(self, account_name: Optional[str]) -> Optional[str]:
        """Resolve and validate account access."""
        account = account_name or self.default_account

        if self.allowed_accounts and account:
            if account not in self.allowed_accounts:
                available = ", ".join(self.allowed_accounts)
                raise ValueError(
                    f"Account '{account}' not allowed. Available accounts: {available}"
                )

        return account

    def _format_messages(
        self, messages: List[Any], account: Optional[str], folder: str
    ) -> str:
        """Format email messages for agent consumption."""
        if not messages:
            return f"No unread emails in {folder}" + (
                f" (account: {account})" if account else ""
            )

        lines = [f"Found {len(messages)} unread email(s) in {folder}:"]
        lines.append("")

        for idx, msg in enumerate(messages, 1):
            date_str = msg.date.strftime("%Y-%m-%d %H:%M") if msg.date else "unknown"
            sender = (
                f"{msg.from_name} <{msg.from_address}>"
                if msg.from_name
                else msg.from_address
            )

            lines.append(f"--- Email {idx} ---")
            lines.append(f"UID: {msg.uid}")
            lines.append(f"Subject: {msg.subject}")
            lines.append(f"From: {sender}")
            lines.append(f"Date: {date_str}")

            if msg.has_attachments:
                att_list = ", ".join(msg.attachment_names[:5])
                lines.append(f"Attachments: {att_list}")

            # Truncate body preview
            body_preview = msg.body_text[:500].strip()
            if len(msg.body_text) > 500:
                body_preview += "... [truncated]"
            lines.append(f"Body:\n{body_preview}")
            lines.append("")

        return "\n".join(lines)


# Valid IMAP SEARCH keywords for query validation
IMAP_SEARCH_KEYWORDS = {
    "ALL",
    "ANSWERED",
    "BCC",
    "BEFORE",
    "BODY",
    "CC",
    "DELETED",
    "DRAFT",
    "FLAGGED",
    "FROM",
    "HEADER",
    "KEYWORD",
    "LARGER",
    "NEW",
    "NOT",
    "OLD",
    "ON",
    "OR",
    "RECENT",
    "SEEN",
    "SENTBEFORE",
    "SENTON",
    "SENTSINCE",
    "SINCE",
    "SMALLER",
    "SUBJECT",
    "TEXT",
    "TO",
    "UID",
    "UNANSWERED",
    "UNDELETED",
    "UNDRAFT",
    "UNFLAGGED",
    "UNKEYWORD",
    "UNSEEN",
}


def normalize_imap_query(query: str) -> str:
    """
    Normalize and validate IMAP search query.

    If query doesn't start with a valid IMAP keyword, wraps it in SUBJECT "..."
    to prevent agent from passing free-form text that causes IMAP errors.

    Also removes non-ASCII characters that would cause encoding errors.
    """
    query = query.strip()
    if not query:
        return "ALL"

    # Check if query starts with a valid IMAP keyword
    first_word = query.split()[0].upper()
    if first_word not in IMAP_SEARCH_KEYWORDS:
        # Agent passed free-form text - wrap in SUBJECT search
        # Remove non-ASCII characters to prevent encoding errors
        safe_query = query.encode("ascii", errors="ignore").decode("ascii").strip()
        if not safe_query:
            # If nothing left after removing non-ASCII, search ALL.
            # Log only metadata — the query text is user content (PII).
            logger.warning(
                "Query (len=%d) contains only non-ASCII characters, defaulting to ALL",
                len(query),
            )
            return "ALL"
        # Escape quotes in the search term
        safe_query = safe_query.replace('"', '\\"')
        logger.info("Normalized free-form query (len=%d) to SUBJECT search", len(query))
        return f'SUBJECT "{safe_query}"'

    # Valid IMAP query - still sanitize non-ASCII in string arguments
    # This is a simplified approach; full parsing would be more complex
    try:
        # Test if it can be encoded as ASCII
        query.encode("ascii")
        return query
    except UnicodeEncodeError:
        # Remove non-ASCII characters
        safe_query = query.encode("ascii", errors="ignore").decode("ascii").strip()
        logger.warning(
            "Removed non-ASCII characters from query: '%s' -> '%s'",
            query[:50],
            safe_query[:50],
        )
        return safe_query if safe_query else "ALL"


class SearchEmailsTool(BaseTool):
    """Tool for searching emails by various criteria.

    Supports IMAP search syntax for flexible queries.
    Automatically normalizes invalid queries to prevent errors.
    """

    name: str = "search_emails"
    description: str = """Search emails in mailbox using IMAP search syntax.
Use when you need to find specific emails.

⚠️ IMPORTANT: The query MUST use valid IMAP search syntax. Do NOT pass free-form text.

Arguments:
  - query (required): IMAP search query. MUST start with a valid keyword:
    - FROM "address@example.com" - emails from specific sender
    - TO "address@example.com" - emails to specific recipient  
    - SUBJECT "keyword" - emails with keyword in subject (ASCII only!)
    - BODY "keyword" - emails with keyword in body (ASCII only!)
    - TEXT "keyword" - search in entire message (ASCII only!)
    - SINCE 01-Jan-2026 - emails since date (format: DD-Mon-YYYY)
    - BEFORE 01-Jan-2026 - emails before date
    - ON 01-Jan-2026 - emails on specific date
    - UNSEEN - unread emails only
    - SEEN - read emails only
    - ALL - all emails
    - Combine with OR/NOT: OR FROM "a@x.com" FROM "b@x.com"

  ⚠️ RESTRICTIONS:
    - Use ASCII characters only in search terms (no Polish: ą,ę,ó,ł etc.)
    - Always quote string arguments: SUBJECT "meeting" not SUBJECT meeting
    - Date format must be DD-Mon-YYYY (e.g., 01-Jan-2026)

  - account_name (optional): Email account alias to use
  - folder (optional): Mailbox folder to search. Default: INBOX
  - limit (optional): Maximum results. Default: 10

Returns list of matching emails with details (UID, sender, subject, date).
"""

    imap_service: Any = Field(exclude=True)
    allowed_accounts: Optional[List[str]] = Field(default=None, exclude=True)
    default_account: Optional[str] = Field(default=None, exclude=True)

    def _run(
        self,
        query: str,
        account_name: Optional[str] = None,
        folder: str = "INBOX",
        limit: int = 10,
    ) -> str:
        try:
            account = self._resolve_account(account_name)

            # Normalize query to prevent IMAP errors from malformed agent input
            normalized_query = normalize_imap_query(query)
            if normalized_query != query:
                # Metadata only — the query text is user content (PII).
                logger.info(
                    "SearchEmailsTool: normalized query (len=%d -> len=%d)",
                    len(query),
                    len(normalized_query),
                )

            messages = self.imap_service.fetch_emails(
                host_alias=account,
                folder=folder,
                limit=limit,
                search_criteria=normalized_query,
            )
            return self._format_search_results(messages, normalized_query, account)

        except ValueError as e:
            logger.error("Search error: %s", e)
            return f"✗ {e}"
        except Exception as e:
            logger.error("Failed to search emails: %s", e, exc_info=True)
            return f"✗ Failed to search emails: {e}"

    async def _arun(
        self,
        query: str,
        account_name: Optional[str] = None,
        folder: str = "INBOX",
        limit: int = 10,
    ) -> str:
        return self._run(query, account_name, folder, limit)

    def _resolve_account(self, account_name: Optional[str]) -> Optional[str]:
        """Resolve and validate account access."""
        account = account_name or self.default_account

        if self.allowed_accounts and account:
            if account not in self.allowed_accounts:
                available = ", ".join(self.allowed_accounts)
                raise ValueError(
                    f"Account '{account}' not allowed. Available: {available}"
                )

        return account

    def _format_search_results(
        self, messages: List[Any], query: str, account: Optional[str]
    ) -> str:
        """Format search results for agent."""
        if not messages:
            return f"No emails found matching: {query}"

        lines = [f"Found {len(messages)} email(s) matching '{query}':"]
        lines.append("")

        for idx, msg in enumerate(messages, 1):
            date_str = msg.date.strftime("%Y-%m-%d %H:%M") if msg.date else "unknown"
            sender = msg.from_address
            read_status = "📖" if msg.is_read else "📬"

            lines.append(
                f"{idx}. UID: {msg.uid} | {read_status} | {date_str} | {sender} | {msg.subject[:60]}"
            )

        return "\n".join(lines)


class GetEmailTool(BaseTool):
    """Tool for getting full content of a specific email.

    Fetches a single email by UID and returns full details including body.
    """

    name: str = "get_email"
    description: str = """Get full content of a specific email by UID.
Use when you need to read the full body of an email found via search or listing.

Arguments:
  - uid (required): The UID of the email to fetch (e.g., from search results)
  - account_name (optional): Email account alias to use
  - folder (optional): Mailbox folder. Default: INBOX

Returns full email details including headers and body.
"""

    imap_service: Any = Field(exclude=True)
    allowed_accounts: Optional[List[str]] = Field(default=None, exclude=True)
    default_account: Optional[str] = Field(default=None, exclude=True)

    def _run(
        self,
        uid: str,
        account_name: Optional[str] = None,
        folder: str = "INBOX",
    ) -> str:
        try:
            account = self._resolve_account(account_name)

            # Validate UID is numeric
            if not uid.isdigit():
                return f"✗ Invalid UID '{uid}'. UID must be a number."

            # Fetch specific UID
            messages = self.imap_service.fetch_emails(
                host_alias=account,
                folder=folder,
                limit=1,
                search_criteria=f"UID {uid}",
            )

            if not messages:
                return (
                    f"✗ Email with UID {uid} not found in {folder} (account: {account})"
                )

            return self._format_message(messages[0])

        except ValueError as e:
            logger.error("Get email error: %s", e)
            return f"✗ {e}"
        except Exception as e:
            logger.error("Failed to get email: %s", e, exc_info=True)
            return f"✗ Failed to get email: {e}"

    async def _arun(
        self,
        uid: str,
        account_name: Optional[str] = None,
        folder: str = "INBOX",
    ) -> str:
        return self._run(uid, account_name, folder)

    def _resolve_account(self, account_name: Optional[str]) -> Optional[str]:
        """Resolve and validate account access."""
        account = account_name or self.default_account

        if self.allowed_accounts and account:
            if account not in self.allowed_accounts:
                available = ", ".join(self.allowed_accounts)
                raise ValueError(
                    f"Account '{account}' not allowed. Available: {available}"
                )

        return account

    def _format_message(self, msg: Any) -> str:
        """Format full email message for agent."""
        lines = [f"--- Email Details [UID: {msg.uid}] ---"]

        date_str = msg.date.strftime("%Y-%m-%d %H:%M:%S") if msg.date else "unknown"
        sender = (
            f"{msg.from_name} <{msg.from_address}>"
            if msg.from_name
            else msg.from_address
        )
        recipients = ", ".join(msg.to_addresses)

        lines.append(f"Subject: {msg.subject}")
        lines.append(f"Date: {date_str}")
        lines.append(f"From: {sender}")
        lines.append(f"To: {recipients}")

        if msg.cc_addresses:
            lines.append(f"CC: {', '.join(msg.cc_addresses)}")

        if msg.is_read:
            lines.append("Status: Read")
        else:
            lines.append("Status: Unread")

        if msg.has_attachments:
            att_list = ", ".join(msg.attachment_names)
            lines.append(f"Attachments: {att_list}")

        lines.append("-" * 30)
        lines.append("Body:")
        lines.append(msg.body_text)

        return "\n".join(lines)


class SendEmailTool(BaseTool):
    """Tool for composing and sending emails.

    Uses existing EmailService (SMTP) for delivery.
    Supports account-specific signatures.
    """

    name: str = "send_email"
    description: str = """Compose and send an email.
Use when you need to reply to or compose a new email.

Arguments:
  - to (required): Recipient email address(es). Can be comma-separated for multiple.
  - subject (required): Email subject line
  - body (required): Email body text
  - account_name (optional): Email account to send from. Determines sender address and signature.
  - cc (optional): CC recipients (comma-separated)
  - reply_to_message_id (optional): Message-ID if this is a reply

The account's configured signature will be automatically appended if available.
"""

    email_service: Any = Field(exclude=True)
    email_config: Any = Field(exclude=True)
    allowed_accounts: Optional[List[str]] = Field(default=None, exclude=True)
    default_account: Optional[str] = Field(default=None, exclude=True)

    def _run(
        self,
        to: str,
        subject: str,
        body: str,
        account_name: Optional[str] = None,
        cc: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
    ) -> str:
        try:
            account = self._resolve_account(account_name)
            host_config = self.email_config.email.get_host_config(account)

            # Append signature if configured
            full_body = body
            if host_config.signature:
                full_body = f"{body}\n\n--\n{host_config.signature}"

            # Parse recipients
            to_list = [addr.strip() for addr in to.split(",") if addr.strip()]
            cc_list = (
                [addr.strip() for addr in cc.split(",") if addr.strip()] if cc else None
            )

            # Build headers for reply threading
            headers = None
            if reply_to_message_id:
                headers = {
                    "In-Reply-To": reply_to_message_id,
                    "References": reply_to_message_id,
                }

            message_id = self.email_service.send_email(
                to=to_list,
                subject=subject,
                text=full_body,
                host_alias=account,
                cc=cc_list,
                headers=headers,
            )

            logger.info(
                "Email sent via tool: to=%s, subject=%s, account=%s",
                to,
                subject[:30],
                account,
            )
            return f"✓ Email sent successfully to {to} (Message-ID: {message_id})"

        except ValueError as e:
            logger.error("Send email error: %s", e)
            return f"✗ {e}"
        except Exception as e:
            logger.error("Failed to send email: %s", e, exc_info=True)
            return f"✗ Failed to send email: {e}"

    async def _arun(
        self,
        to: str,
        subject: str,
        body: str,
        account_name: Optional[str] = None,
        cc: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
    ) -> str:
        return self._run(to, subject, body, account_name, cc, reply_to_message_id)

    def _resolve_account(self, account_name: Optional[str]) -> Optional[str]:
        """Resolve and validate account access."""
        account = account_name or self.default_account

        if self.allowed_accounts and account:
            if account not in self.allowed_accounts:
                available = ", ".join(self.allowed_accounts)
                raise ValueError(
                    f"Account '{account}' not allowed. Available: {available}"
                )

        return account


class SummarizeEmailTool(BaseTool):
    """Tool for summarizing email content using LLM.

    Classifies email type and extracts key information.
    Useful for triaging incoming emails.
    """

    name: str = "summarize_email"
    description: str = f"""Summarize and classify an email.
Use to quickly understand email content and determine if action is needed.

Arguments:
  - subject (required): Email subject line
  - body (required): Email body text
  - from_address (optional): Sender address for context

Returns:
  - summary: Concise summary of the email
  - email_type: Classification ({', '.join(EMAIL_TYPES)})
  - requires_action: Whether response/action is needed
  - key_points: Main points from the email
"""

    model_name: str = Field(default="gpt-5-nano", exclude=True)

    def _run(
        self,
        subject: str,
        body: str,
        from_address: Optional[str] = None,
    ) -> str:
        try:
            from langchain.chat_models import init_chat_model
            from langchain.messages import HumanMessage, SystemMessage

            model = init_chat_model(self.model_name)
            model_structured = model.with_structured_output(EmailSummary)

            system_prompt = """You are an email analysis assistant.
Analyze the provided email and return a structured summary.

Classification guidelines:
- spam: Unsolicited promotional content, phishing attempts
- newsletter: Regular subscriptions, updates, digests
- important: Work-related, requiring response or action
- transactional: Receipts, confirmations, notifications
- personal: Direct personal communications

Be concise but capture essential information."""

            content = f"Subject: {subject}\n"
            if from_address:
                content += f"From: {from_address}\n"
            content += f"\nBody:\n{body[:4000]}"  # Limit body length

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ]

            result: EmailSummary = model_structured.invoke(messages)

            # Format response
            lines = [
                f"📧 Email Summary",
                f"Type: {result.email_type.value}",
                f"Requires Action: {'Yes ⚠️' if result.requires_action else 'No'}",
                f"",
                f"Summary: {result.summary}",
            ]

            if result.key_points:
                lines.append("")
                lines.append("Key Points:")
                for point in result.key_points:
                    lines.append(f"  • {point}")

            return "\n".join(lines)

        except Exception as e:
            logger.error("Failed to summarize email: %s", e, exc_info=True)
            return f"✗ Failed to summarize email: {e}"

    async def _arun(
        self,
        subject: str,
        body: str,
        from_address: Optional[str] = None,
    ) -> str:
        return self._run(subject, body, from_address)


class ListEmailAccountsTool(BaseTool):
    """Tool for listing available email accounts.

    Shows which accounts the agent can access.
    """

    name: str = "list_email_accounts"
    description: str = """List available email accounts.
Use to see which email accounts you can read from or send to.

No arguments required.

Returns list of account names with their capabilities (IMAP read, SMTP send).
"""

    email_config: Any = Field(exclude=True)
    imap_service: Any = Field(default=None, exclude=True)
    allowed_accounts: Optional[List[str]] = Field(default=None, exclude=True)

    def _run(self) -> str:
        try:
            all_hosts = self.email_config.email.list_host_aliases()
            imap_hosts = (
                self.imap_service.list_configured_hosts() if self.imap_service else []
            )

            # Filter by allowed accounts if configured
            if self.allowed_accounts:
                all_hosts = [h for h in all_hosts if h in self.allowed_accounts]

            if not all_hosts:
                return "No email accounts available."

            lines = ["Available email accounts:"]
            for host in all_hosts:
                config = self.email_config.email.get_host_config(host)
                capabilities = ["📤 Send (SMTP)"]
                if host in imap_hosts:
                    capabilities.append("📥 Read (IMAP)")

                caps_str = ", ".join(capabilities)
                context = f" - {config.context}" if config.context else ""
                lines.append(f"  • {host}: {caps_str}{context}")

            default = self.email_config.email.default_host
            if default in all_hosts:
                lines.append(f"\nDefault account: {default}")

            return "\n".join(lines)

        except Exception as e:
            logger.error("Failed to list accounts: %s", e)
            return f"✗ Failed to list accounts: {e}"

    async def _arun(self) -> str:
        return self._run()


def get_email_tools(
    email_service: Optional["EmailService"] = None,
    imap_service: Optional["ImapService"] = None,
    allowed_accounts: Optional[List[str]] = None,
    default_account: Optional[str] = None,
    include_summarize: bool = True,
    summarize_model: str = "gpt-5-nano",
) -> List[BaseTool]:
    """Factory for email tools to ease agent wiring.

    Creates configured email tools based on available services.

    Args:
        email_service: EmailService instance for sending (optional)
        imap_service: ImapService instance for reading (optional)
        allowed_accounts: List of account aliases the agent can access.
                          If None, all configured accounts are allowed.
        default_account: Default account to use when not specified.
        include_summarize: Whether to include SummarizeEmailTool.
        summarize_model: Model to use for email summarization.

    Returns:
        List of configured BaseTool instances.

    Example:
        from inference_core.services.email_service import get_email_service
        from inference_core.services.imap_service import get_imap_service

        tools = get_email_tools(
            email_service=get_email_service(),
            imap_service=get_imap_service(),
            allowed_accounts=['primary', 'support'],
            default_account='primary',
        )
    """
    tools: List[BaseTool] = []

    # Get email config for shared configuration
    try:
        from inference_core.core.email_config import get_email_config

        email_config = get_email_config()
    except Exception as e:
        logger.warning("Could not load email config: %s", e)
        return tools

    # Add IMAP tools if service available
    if imap_service:
        tools.append(
            ReadUnseenEmailsTool(
                imap_service=imap_service,
                allowed_accounts=allowed_accounts,
                default_account=default_account,
            )
        )
        tools.append(
            SearchEmailsTool(
                imap_service=imap_service,
                allowed_accounts=allowed_accounts,
                default_account=default_account,
            )
        )
        tools.append(
            GetEmailTool(
                imap_service=imap_service,
                allowed_accounts=allowed_accounts,
                default_account=default_account,
            )
        )

    # Add SMTP tool if service available
    if email_service:
        tools.append(
            SendEmailTool(
                email_service=email_service,
                email_config=email_config,
                allowed_accounts=allowed_accounts,
                default_account=default_account,
            )
        )

    # Add account listing tool
    tools.append(
        ListEmailAccountsTool(
            email_config=email_config,
            imap_service=imap_service,
            allowed_accounts=allowed_accounts,
        )
    )

    # Add summarization tool
    if include_summarize:
        tools.append(SummarizeEmailTool(model_name=summarize_model))

    logger.info(
        "Created %d email tools (IMAP: %s, SMTP: %s, accounts: %s)",
        len(tools),
        imap_service is not None,
        email_service is not None,
        allowed_accounts or "all",
    )

    return tools


def generate_email_tools_system_instructions() -> str:
    """Generate system prompt instructions for email tools usage.

    Returns formatted instructions for agent system prompts.
    """
    return """## Email Tools Usage

You have access to email tools for reading and sending emails:

### list_email_accounts
Use this first to see which email accounts you can access.
Shows account names, capabilities (read/send), and context.

### read_unseen_emails
Fetch unread emails from a mailbox.
**When to use:**
- Checking for new emails
- Reviewing inbox at start of conversation
- Monitoring for important messages

### get_email
Get full content of a specific email by UID.
**When to use:**
- Reading the full body of an email found via search
- When you need details not shown in the summary list
- Retrieving attachments info and full headers

### search_emails
Search emails by various criteria.
**When to use:**
- Finding specific emails by sender, subject, date
- Looking for emails about a topic
- Retrieving email by UID for detailed view

**Search examples:**
- `FROM sender@example.com` - emails from sender
- `SUBJECT urgent` - emails with 'urgent' in subject
- `SINCE 01-Jan-2026` - emails after date
- `UNSEEN FROM boss@company.com` - unread from specific sender

### send_email
Compose and send emails.
**When to use:**
- Replying to emails
- Sending new messages
- Following up on requests

**Important:**
- Always confirm email content with user before sending
- Include reply_to_message_id when replying to maintain threading
- Account signature is automatically appended

### summarize_email
Quickly analyze email content.
**When to use:**
- Triaging multiple emails
- Determining if action is needed
- Extracting key points from long emails

**Best practices:**
1. Use list_email_accounts to understand available accounts
2. Check for new emails with read_unseen_emails
3. Summarize important emails to understand context
4. Confirm with user before sending any replies
5. Use search_emails to find specific messages when needed
"""


__all__ = [
    "ReadUnseenEmailsTool",
    "SearchEmailsTool",
    "GetEmailTool",
    "SendEmailTool",
    "SummarizeEmailTool",
    "ListEmailAccountsTool",
    "EmailSummary",
    "EmailType",
    "get_email_tools",
    "generate_email_tools_system_instructions",
]
