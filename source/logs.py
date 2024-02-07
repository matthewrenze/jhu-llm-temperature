# Import libraries
from enum import Enum
from colorama import Fore, Style, init

# Initialize colorama for automatic style reset
init(autoreset=True)

# Define the log levels
class LogLevel(Enum):
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4,
    HEAD = 5

# Define the ANSI colors
BOLD_WHITE = "\033[97m"
BOLD_YELLOW = "\033[93;1m"
BOLD_RED = "\033[91;1m"
WHITE = "\033[38;5;250m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# TESTING
ORANGE = "\033[38;5;208m"
BOLD_ORANGE = "\033[38;5;208;1m"
GREY = "\033[38;5;240m"
BOLD_GREY = "\033[38;5;245;1m"

# Define the keywords to bold
keywords = [
    "Task:",
    "Topic:",
    "Question:",
    "Choices:",
    "Response 1:",
    "Response 2:",
    "Response 3:",
    "Response 4:",
    "Response 5:",
    "Response 6:",
    "Response 7:",
    "Response 8:",
    "Response 9:",
    "Response 10:",
    "Knowledge:",
    "Plan:",
    "Thought:",
    "Criticism:",
    "Votes:",
    "Action:",
    "Observation:",
    "Votes:",
    "Agent Answer:",
    "Correct Answer:",
    "Score:",
    "Steps:",
    "Tokens:",
    "Cost:"]

class Log:
    def __init__(self, log_file_path: str, log_level: LogLevel):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, "a", encoding="utf-8")
        self.log_level = log_level

    def format(self, message):

        # Format headings
        if message.startswith("###"):
            return f"{BOLD_WHITE}{message}{RESET}"

        # Format keywords in the message
        if not message.startswith("Start Time:"):
            for keyword in keywords:
                if keyword in message:
                    message = message.replace(keyword, f"{BOLD_WHITE}{keyword}{RESET}")

        # Format debug messages as gray
        if message.startswith("Debug:"):
            return f"{BOLD_GREY}Debug:{GREY}{message[len('Debug:'):]}{RESET}"

        # Format warnings
        if message.startswith("Warning:"):
            return f"{BOLD_ORANGE}Warning:{BOLD_ORANGE}{message[len('Warning:'):]}{RESET}"

        # Format warnings
        if message.startswith("Error:"):
            return f"{BOLD_RED}Error:{RED}{message[len('Error:'):]}{RESET}"

        # No keyword or special case found, apply regular formatting
        return f"{message}{RESET}"

    def log(self, message):
        message = message.rstrip('\n')
        self.log_file.write(message + '\n')
        message = self.format(message)
        print(message)

    def head(self, message):
        if not message.startswith("###"):
            message = "### " + message + " ###"
        self.log(message)

    def debug(self, message):
        if self.log_level.value > LogLevel.DEBUG.value:
            return

        if not message.startswith("Debug:"):
            message = "Debug: " + message
        self.log(message)

    def info(self, message):
        if self.log_level.value > LogLevel.INFO.value:
            return

        self.log(message)

    def warning(self, message):
        if self.log_level.value > LogLevel.WARNING.value:
            return

        if not message.startswith("Warning:"):
            message = "Warning: " + message
        self.log(message)

    def error(self, message):
        if self.log_level.value > LogLevel.ERROR.value:
            return

        if not message.startswith("Error:"):
            message = "Error: " + message
        self.log(message)

    def close(self):
        self.log_file.flush()
        self.log_file.close()


# # DEBUG: Test the log class
# log = Log("log_test.txt")
# log.head("### Heading ###")
# log.head("Another Heading")
# log.debug("Debug: Some debug message")
# log.debug("Another debug message")
# log.info("Observation: Some observation")
# log.info("Thought: Some thought")
# log.info("Action: Some action")
# log.warning("Warning: Some warning")
# log.warning("Another warning")
# log.info("Just some words")
# log.error("Error: Some error")
# log.error("Another error")
# log.close()

