import time
import sys
import warnings

YELLOW = '\033[93m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

class Logger:
    _instance : 'Logger' = None
    info = '[INFO]'
    warning = '[WARNING]'
    error = '[ERROR]'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._logs = []
            cls._instance._init_time = time.time()
            cls._last_prc = 0
        return cls._instance

    def log(self, message: str, level: str = info):
        formatted = self._formate(message, level)
        self._logs.append(formatted)
        print(formatted + "\n", end= "")
    
    def log_prc(self, message: str, current: float,
                max: float, level: str = info):
        prc = int(100*current/max)
        if prc != self._last_prc:
            formatted = self._formate(f"{message} {prc}%", level)
            self._logs.append(formatted)
            print(formatted + '\r', end = '')
            self._last_prc = prc

    def log_prc_done(self, message: str, level: str = info):
        formatted = self._formate(message + f"{GREEN} 100%{RESET}", level)
        print(formatted + "\n", end= "")
        self._logs.append(formatted)

    def _formate(self, message: str, level: str = info) -> str:
        current_time = time.time()
        elapsed = current_time - self._init_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        if (level == Logger.warning):
            return f"[{formatted_time}]{YELLOW} {self.warning} {RESET}{message}"
        elif (level == Logger.error):
            return f"[{formatted_time}]{RED} {self.error} {message} {RESET}"
        return f"[{formatted_time}] {self.info} {message}                   "

    def get_logs(self):
        return self._logs.copy()

    def raise_error(self, message: str):
        raise InternalError(self._formate(message, self.error))
    
    def ask_for_continue(self, message: str):
        input(self._formate(f"{YELLOW}{message} Continue ? "
                            f"[Press ENTER or use ctrl C] {RESET}",
                             self.warning))


class InternalError(Exception):
    """Custom exception class for errors in this package."""
    def __init__(self, message: str):
        super().__init__(f"\n{RED}{message}{RESET}")


class FilteredOutput:

    def __init__(self, forbidden_keywords: list[str] = []):
        self.forbidden_keywords = forbidden_keywords
        self.original_stdout = sys.stdout # Save the original stdout
        self._number_of_erased_entry = 0
    
    @staticmethod
    def on():
        """Initialize the filtered output."""
        if (sys.stdout is not None and isinstance(sys.stdout, FilteredOutput)):
            Logger.log("FilteredOutput was already active.", Logger.warning)
            return
        warnings.simplefilter("always")  # Ensure all warnings are caught
        # Redirecting warnings to a the standard output
        warnings.showwarning = lambda message, category, filename, lineno,\
              file=None, line=None: \
                 print(f"{category.__name__}: {message}")
        # Redirecting stdout to the FilteredOutput class
        sys.stdout = FilteredOutput(forbidden_keywords=["RuntimeWarning",
                                                        "IntegrationWarning"])

    @staticmethod
    def off():
        """Close the filtered output."""
        if (sys.stdout is not None and isinstance(sys.stdout, FilteredOutput)):
            sys.stdout.print_number_of_warnings()
            sys.stdout.restore_stdout()
        else:
            Logger.log("FilteredOutput was not active.", Logger.warning)

    def write(self, message: str):
        # Ignore messages containing forbidden keywords or empty lines
        if message == "\n":  # Ignore empty messages
            return
        if any(keyword in message for keyword in self.forbidden_keywords):
            self._number_of_erased_entry += 1
            return
        self.original_stdout.write(message)  # Write allowed messages

    def flush(self):
        self.original_stdout.flush()
    
    def print_number_of_warnings(self):
        print(f"During the procedure, {self._number_of_erased_entry} "
              "warnings were erased. They were likely "
              "raised during integrations, nothing to worry about.\n")
    
    def restore_stdout(self):
        sys.stdout = self.original_stdout