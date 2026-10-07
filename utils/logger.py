import gradio as gr
import inspect
from typing import Callable, Any, Union
from joblib import hash as jhash
import asyncio
import threading
import time
import sqlite3
import zlib
import re
import os
from pathlib import Path
from tqdm import tqdm
import logging
from logging import handlers
import rtoml
import json
from functools import wraps
from platformdirs import user_cache_dir, user_log_dir

try:
    from .typechecker import optional_typecheck
    from .shared_module import shared
except ImportError as err:
    if "attempted relative import with no known parent package" not in str(err):
        raise
    else:
        # needed when calling audio_splitter instead of Voice2Anki
        from typechecker import optional_typecheck
        from shared_module import shared

cache_dir = Path(user_cache_dir(appname="Voice2Anki"))
assert cache_dir.parent.exists() or cache_dir.parent.parent.exists(
), f"Invalid cache dir location: '{cache_dir}'"
cache_dir.mkdir(parents=True, exist_ok=True)

log_dir = Path(user_log_dir(appname="Voice2Anki"))
assert log_dir.parent.exists() or log_dir.parent.parent.exists(
) or log_dir.parent.parent.parent.exists(), f"Invalid log_dir location: '{log_dir}'"
log_dir.mkdir(exist_ok=True, parents=True)
log_file = (log_dir / "logs.txt")
log_file.touch(exist_ok=True)

log_formatter = logging.Formatter(
        fmt='%(asctime)s ##%(levelname)s %(funcName)s(%(lineno)d)## %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
file_handler = handlers.RotatingFileHandler(
        log_file,
        mode='a',
        maxBytes=1000000,
        backupCount=100,
        encoding=None,
        delay=0,
        )
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

# the file handler lives on the root logger so that third-party INFO
# records (litellm/openai HTTP chatter) also land in the file, where
# get_log reads them (it already filters the noisy ones out)
root_log = logging.getLogger()
root_log.setLevel(logging.INFO)
root_log.addHandler(file_handler)

# everything the app logs itself flows through this named logger so the
# level of app messages can be controlled without touching third-party
# logs: standard levels by default, DEBUG with --debug (see enable_debug)
app_logger = logging.getLogger("Voice2Anki")
app_logger.setLevel(logging.INFO)
log_regex = re.compile(" ##.*?##")

mess = f"This is a test to check wether the logging works: {time.time()}"
root_log.info(mess)
assert mess in log_file.read_text().splitlines()[-1], "Logs does not appear to be working"

colors = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "reset": "\033[0m",
    "white": "\033[0m",
    "purple": "\033[95m",
    "italic": "\033[3m",
    "bold": "\033[1m",
    "underline": "\033[4m",
    "blink": "\033[5m",

    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
    "bg_white": "\033[47m",
}
# make highvisions more visible
colors["very_high_vis"] = colors["bold"] + colors["underline"] + colors["italic"] + colors["red"] + colors["blink"] + colors["bg_cyan"]
colors["high_vis"] = colors["bold"] + colors["underline"] + colors["italic"] + colors["red"] + colors["bg_cyan"]


@optional_typecheck
def Critical(func: Callable) -> Callable:
    """if this decorator is used, any exception in the wrapped function
    will add to the error message to restart the app.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            gr.Warning(red(f"CRITICAL ERROR - PLEASE RESTART THE APP\nFunction: {func}\nError: {err}"))
            raise
    return wrapper


@optional_typecheck
@Critical
def store_to_db(dictionnary: dict, db_name: str) -> bool:
    """
    take a dictionnary and add it to the sqlite db. This is used to store
    all interactions with LLM and can be used later to create a dataset for
    finetuning.
    """

    Path("databases").mkdir(exist_ok=True)
    data = zlib.compress(
            json.dumps(dictionnary, ensure_ascii=False).encode(),
            level=9,  # 1: fast but large, 9 slow but small
            )
    with shared.db_lock:
        conn = sqlite3.connect(f"./databases/{db_name}.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS dictionaries
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          data TEXT)''')
        cursor.execute("INSERT INTO dictionaries (data) VALUES (?)", (data,))
        conn.commit()
        conn.close()
    return True


@optional_typecheck
def print_db(db_filename: str) -> str:
    Path("databases").mkdir(exist_ok=True)
    assert Path(f"./databases/{db_filename}").exists(), (
        f"db not found: '{db_filename}'")
    conn = sqlite3.connect(f"./databases/{db_filename}")
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM dictionaries")
    rows = cursor.fetchall()
    dictionaries = []
    for row in rows:
        dictionary = json.loads(zlib.decompress(row[0]))
        dictionaries.append(dictionary)
    return json.dumps(dictionaries, ensure_ascii=False, indent=4)


@optional_typecheck
def get_coloured_logger(color_asked: str, level: int = logging.INFO) -> Callable:
    """build a printer that logs through the standard logging module at the
    given level: the message reaches the rotating file log with a real level
    and is printed to the console with its color only while that level is
    enabled. Returns its input unchanged so it can be used inline, e.g.
    `return red("Too few words...")`.
    """
    col = colors[color_asked]

    def printer(string: Any, **args) -> Any:
        inp = string
        if isinstance(string, dict):
            try:
                string = rtoml.dumps(string, pretty=True)
            except Exception:
                string = json.dumps(string, indent=2)
        if isinstance(string, list):
            try:
                string = ",".join(string)
            except:
                pass
        try:
            string = str(string)
        except:
            try:
                string = string.__str__()
            except:
                string = string.__repr__()
        # a real level in the file log through the standard logging module
        app_logger.log(level, string)
        if app_logger.isEnabledFor(level):
            # routed through tqdm so that progress bars are not torn apart
            tqdm.write(col + string + colors["reset"], **args)
        return inp
    return printer

@optional_typecheck
def get_log() -> str:
    "frequently called: read the most recent log entries and display it in the output field"
    global last_log_content, latest_tail
    logcontent = []
    # updates only if the last line has changed
    with open(str(log_file), "rb") as f:
        # source: https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        lastline = f.readline().decode().strip()
        lastline = re.sub(log_regex, " >        ", lastline)[11:]
        if last_log_content and (lastline[23:] == latest_tail[23:] or "HTTP Request: POST" in lastline):
            return last_log_content

    latest_tail = lastline
    with open(str(log_file), "r") as f:
        for line in f.readlines()[-100:]:
            line = line.strip()
            if "HTTP Request: POST" in line:
                continue
            if not line:
                continue
            line = re.sub(log_regex, " >        ", line)[11:]
            logcontent.append(line)
    if not logcontent:
        return "Empty log"
    logcontent.reverse()
    last_log_content = "\n".join(logcontent)
    return last_log_content

latest_tail = None
last_log_content = None


whi = get_coloured_logger("white", level=logging.INFO)
yel = get_coloured_logger("yellow", level=logging.WARNING)
red = get_coloured_logger("red", level=logging.ERROR)
purp = get_coloured_logger("purple")
ital = get_coloured_logger("italic")
bold = get_coloured_logger("bold")
underline = get_coloured_logger("underline")
high_vis = get_coloured_logger("high_vis")
very_high_vis = get_coloured_logger("very_high_vis")
# tracing-style diagnostics: silent on the console and in the file until
# enable_debug() lowers the level to DEBUG (--debug)
deb = get_coloured_logger("italic", level=logging.DEBUG)


@optional_typecheck
def enable_debug() -> None:
    """--debug: lower the app logger and the file handler to DEBUG so that
    deb() diagnostics show up on the console and in the file log."""
    app_logger.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)


@optional_typecheck
def Timeout(limit: Union[int, Callable[[], int]]) -> Callable:
    """wrapper to add a timeout to function. I had to use threading because
    signal could not be used outside of the main thread in gradio

    ``limit`` may be a callable, evaluated on each call, for functions whose
    acceptable duration depends on what they are about to do -- a local 3B
    model on CPU needs minutes where a cloud API needs seconds.
    """
    if shared.disable_timeout:
        @optional_typecheck
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

    def resolve_limit() -> int:
        return limit() if callable(limit) else limit

    @optional_typecheck
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                async with asyncio.timeout(resolve_limit()):
                    return await func(*args, **kwargs)
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # return func(*args, **kwargs)  # for debugging
                seconds = resolve_limit()
                result = []
                def appender(func, *args, **kwargs):
                    result.append(func(*args, **kwargs))
                thread = threading.Thread(
                        target=appender,
                        args=[func] + list(args),
                        kwargs=kwargs,
                        daemon=False,
                        )
                thread.start()

                # add the thread in the shared module, this way we can empty
                # the list to cut the timeout short
                with shared.timeout_lock:
                    shared.running_threads["timeout"].append(thread)

                start = time.time()
                while shared.running_threads["timeout"] and thread.is_alive():
                    time.sleep(0.1)
                    if time.time() - start > seconds:
                        raise Exception(f"Reached timeout for {func} after {seconds}s")
                if not shared.running_threads["timeout"]:
                    raise Exception(f"Thread of func {func} was killed")

                if not result:  # meaning an exception occured in the function
                    raise Exception(f"No result from {func} with args {args} {kwargs}")
                else:
                    return result[0]
        return wrapper
    return decorator

@optional_typecheck
def smartcache(func: Callable) -> Callable:
    """used to decorate a function that is already decorated by a
    joblib.Memory decorator. It stores the hash of the arguments in
    shared.smartcache at the start of the run and removes it at the end.
    If it already exists that means the cache is already computing the same
    value so just wait for that to finish to avoid concurrent calls."""
    if shared.disable_smartcache:
        @optional_typecheck
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

    @wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(func, "check_call_in_cache"):
            f = func.func
            # ignored arguments to take into account
            kwargs2 = kwargs.copy()
            for ig in func.ignore:
                if ig in kwargs2:
                    del kwargs2[ig]
        else:
            f = func
            kwargs2 = kwargs.copy()

        kwargs_sorted = {}
        for k in sorted(kwargs2.keys()):
            kwargs_sorted[k] = kwargs2[k]

        # ORACLE C1 FIX: build a SECOND sorted dict from the *unsorted* full
        # kwargs for the actual invocation -- ``kwargs_sorted`` above has
        # the joblib-``ignore`` entries stripped, and using it as the call
        # dict means e.g. ``audio_edit`` (which passes ``audio_path=`` to
        # ``whisper_cached``, decorated ``@stt_cache.cache(ignore=["audio_path"])``)
        # silently loses that kwarg and raises ``ValueError: Wrong number of
        # arguments``. Hash with the stripped dict; call with the full set.
        kwargs_full_sorted = {k: kwargs[k] for k in sorted(kwargs.keys())}

        fstr = str(f)

        to_hash = [inspect.getsource(f), fstr]
        if args:
            to_hash.append(args)
        if kwargs_sorted:
            to_hash.append(kwargs_sorted)
        h = jhash(to_hash)

        if h in shared.smartcache:
            t = shared.smartcache[h]
            red(f"Smartcache: already ongoing for {fstr} since {time.time()-t:.2f}s: hash={h}")
            i = 0
            while h in shared.smartcache:
                time.sleep(0.1)
                i += 1
                if i % 10 == 0:
                    delay = time.time() - t
                    red(f"Smartcache: waiting for {fstr} caching to finish for {delay:.2f}s: hash={h}")
            if hasattr(func, "check_call_in_cache") and not func.check_call_in_cache(*args, **kwargs):
                red(f"Smartcache: after waiting for {fstr} the result was still missing from the cache.")
            return func(*args, **kwargs)

        else:
            with shared.thread_lock:
                with shared.timeout_lock:
                    shared.smartcache[h] = time.time()
            try:
                if args:
                    result = func(*args, **kwargs_full_sorted)
                else:
                    result = func(**kwargs_full_sorted)
            except:
                with shared.thread_lock:
                    with shared.timeout_lock:
                        del shared.smartcache[h]
                raise
            if h in shared.smartcache:
                with shared.thread_lock:
                    with shared.timeout_lock:
                        del shared.smartcache[h]
            return result
    return wrapper
