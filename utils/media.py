import os
import shutil
import re
import shutil
import time
from joblib import Parallel, delayed
from typing import List, Union, Tuple, Optional
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from array import array
from scipy.signal import butter, sosfilt
from pathlib import Path, PosixPath
import gradio as gr
import queue
import pickle
from bs4 import BeautifulSoup
import cv2
import numpy as np
import pyclip
import hashlib
import copy
from PIL import Image

from .logger import whi, red, Timeout
from .ocr import get_text
from .shared_module import shared
from .typechecker import optional_typecheck


@optional_typecheck
def is_image_pillow(file_path: Union[str, PosixPath]) -> bool:
    """Header-only check via Pillow, so no libmagic system library is needed."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

@optional_typecheck
def is_image_cv2(file_path: Union[str, PosixPath]) -> bool:
    try:
        # Attempt to read the image header
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        return img is not None
    except cv2.error:
        return False

@optional_typecheck
def get_image(gallery) -> Optional[List[Union[gr.Gallery, np.ndarray]]]:
    whi("Getting image from clipboard")
    assert shared.pv["enable_gallery"], "Incoherent UI"
    orig_gallery = copy.deepcopy(gallery)
    if hasattr(gallery, "root"):
        gallery = gallery.root
    try:
        # load from clipboard
        try:
            pasted = pyclip.paste()
        except Exception as err:
            if "DISPLAY" not in os.environ:
                gr.Warning(red(f"Error when getting clipboard content. Maybe try to set the DISPLAY env variable.\nError was: '{err}'"))
            else:
                gr.Warning(red(f"Error when getting clipboardcontent: '{err}'"))
            return orig_gallery

        try:
            pasted_str = pasted.decode().strip()
            pasted_path = Path(pasted_str)
            if pasted_path.exists():
                pasted = pasted_str
            else:
                red(f"pasted: {pasted}")
        except Exception as e:
            red(e)
        if isinstance(pasted, str):
            red(f"Received str from clipboard: {pasted}")
            path = Path(pasted)
            assert path.exists(), f"Pasted string but not a path: {path}"
            if path.is_dir():
                assert gallery is None, "If pasting path to a dir, the target gallery must be None"
                files = sorted([f for f in path.iterdir() if f.suffix.lower()[1:] in ["png", "jpg", "jpeg"]])
                assert files, f"No files ending in png, jpg or jpeg found in {path}"
                files = sorted(files, key=lambda f: f.stat().st_ctime)
                red(f"Will paste those files: {files}")
                images = [rgb_to_bgr(cv2.imread(f.resolve().absolute().__str__(), flags=1)) for f in files]
                red("Done loading those images")
                return images
            elif path.is_file():
                assert path.suffix.lower()[1:] in ["png", "jpg", "jpeg"], f"Only expecting path to files that end in png, jpg or jpeg. Received {path}"
                decoded = rgb_to_bgr(cv2.imread(path.resolve().absolute().__str__(), flags=1))
            else:
                gr.Warning(red(f"Unexpected type of path: {path}"))
                return orig_gallery
        elif isinstance(pasted, bytes):
            decoded = cv2.imdecode(np.frombuffer(pasted, np.uint8), flags=1)
            decoded = rgb_to_bgr(decoded)
        else:
            gr.Warning(red(f"Unexpected type of pasted: {pasted}"))
            return orig_gallery

        if decoded is None:
            whi("Image from clipboard was Nonetype")
            return gallery

        if gallery is None:
            return [decoded]
        if not isinstance(gallery, list):
            red(f'gallery is not list or None but {type(gallery)}')
            return None

        out = []
        for im in gallery:
            if isinstance(im, tuple):
                assert Path(im[0]).exists(), f"Missing image from tuple {im}"
                assert im[1] is None, f"Unexpected tupe: {im}"
                out.append(
                        rgb_to_bgr(
                            cv2.imread(
                                im[0],
                                flags=1)
                            )
                        )
            else:
                out.append(
                        rgb_to_bgr(
                            cv2.imread(
                                im.image.path,
                                flags=1)
                            )
                        )
        out += [decoded]

        whi("Loaded image from clipboard.")
        return out

    except Exception as err:
        gr.Warning(red(f"Error: {err}"))
        return orig_gallery


@optional_typecheck
def check_source(source: str) -> str:
    "makes sure the source is only an img"
    whi("Checking source")
    assert shared.pv["enable_gallery"], "Incoherent UI"
    if source:
        soup = BeautifulSoup(source, 'html.parser')
        imgs = soup.find_all("img")
        source = "</br>".join([str(x) for x in imgs])
        assert source, f"invalid source: {source}"
        # save for next startup
        with open("./cache/voice_cards_last_source.pickle", "wb") as f:
            pickle.dump(source, f)
    else:
        source = ""
    return source


#@Timeout(120)
@optional_typecheck
def get_img_source(gallery: Union[List, None], queue=queue.Queue(), use_html: bool = True) -> None:
    whi("Getting source from image")
    assert shared.pv["enable_gallery"], "Incoherent UI"

    try:
        if hasattr(gallery, "root"):
            gallery = gallery.root
        assert isinstance(gallery, (type(None), list)), "Gallery is not a list or None"
        if gallery is None:
            return queue.put(red("No image in gallery."))
        if len(gallery) == 0:
            return queue.put(red("0 image found in gallery."))

        paths = []
        for img in gallery:
            try:
                path = img.image.path
                assert is_image_pillow(path) or is_image_cv2(path), f"Not an image: {path}"
            except Exception:
                try:
                    path = img["image"]["path"]
                    if path.startswith("http"):
                        path = path.split("file=")[1]
                    cnt = 0
                    while not Path(path).exists():
                        time.sleep(0.1)
                        cnt += 1
                        if cnt == 10:
                            raise Exception(f"img not found in path: {path}")
                    assert is_image_pillow(path) or is_image_cv2(path), f"Not an image: {path}"
                except Exception:
                    # must be a tuple
                    assert isinstance(img, tuple), f"Invalid img type: {img}"
                    assert len(img) == 2, f"Invalid img: {img}"
                    assert img[1] is None, f"Invalid img: {img}"
                    cnt = 0
                    while not Path(img[0]).exists():
                        time.sleep(0.1)
                        cnt += 1
                        if cnt == 10:
                            raise Exception(f"img not found: {img[0]}")
                    assert is_image_pillow(path) or is_image_cv2(path), f"Not an image: {path}"
            img_hash = hashlib.md5(open(path, 'rb').read()).hexdigest()[:10]
            new = shared.anki_media / f"{img_hash}.png"
            if not new.exists():
                shutil.copy2(str(path), str(new))
            assert new.exists(), new
            if new in paths:
                red(f"There's at least one image appearing multiple times in the gallery. Ignoring duplicates: {new}")
            else:
                paths.append(new)

        ocrs = Parallel(
            n_jobs=2,
            backend="threading",
        )(delayed(get_text)(str(new))
            for new in paths
        )

        source = ""
        for path, ocr in zip(paths, ocrs):

            if use_html:
                if ocr:
                    ocr = ocr.replace("\"", "").replace("'", "")
                    ocr = f"title=\"{ocr}\" "

                newsource = f'<img src="{path.name}" {ocr}type="made_by_Voice2Anki">'

                # only add if not duplicate, somehow
                if newsource not in source:
                    source += newsource

                source = check_source(source)
            else:
                source += "\n" + ocr

        return queue.put(source)
    except Exception as err:
        return queue.put(red(f"Error getting source: '{err}' from {gallery}"))

@optional_typecheck
def ocr_image(gallery: Union[List, None]) -> None:
    "use OCR to get the text of an image to display in a textbox"
    q = queue.Queue()
    get_img_source(gallery, q, use_html=False)
    return q.get()


@optional_typecheck
def reset_gallery() -> None:
    whi("Reset images.")
    shared.pv["gallery"] = None
    assert shared.pv["enable_gallery"], "Incoherent UI"


@optional_typecheck
def reset_audio() -> List[dict]:
    whi("Resetting all audio")
    return [gr.update(value=None, label=f"Audio #{i+1}") for i in range(shared.audio_slot_nb)]

def _sox_norm(seg: AudioSegment) -> AudioSegment:
    """SoX `norm`: scale so the loudest sample reaches 0 dBFS."""
    peak = seg.max_dBFS
    if np.isneginf(peak):
        return seg
    return seg.apply_gain(-peak)


def _butter_filter(seg: AudioSegment, btype: str, cutoff_hz: float, order: int) -> AudioSegment:
    """ Butterworth filter on every channel, mirroring SoX's highpass/lowpass. """
    sr = seg.frame_rate
    cutoff = min(max(float(cutoff_hz), 1.0), (sr / 2) * 0.999)
    sos = butter(order, cutoff, btype=btype, fs=sr, output="sos")
    typecode = {1: "b", 2: "h", 4: "l"}[seg.sample_width]
    lo, hi = -(2 ** (8 * seg.sample_width - 1)), 2 ** (8 * seg.sample_width - 1) - 1
    channels = [seg] if seg.channels == 1 else seg.split()
    filtered = []
    for ch in channels:
        samples = np.array(ch.get_array_of_samples(), dtype=np.float64)
        vals = np.clip(np.round(sosfilt(sos, samples)), lo, hi).astype(np.int64)
        filtered.append(vals)
    interleaved = np.stack(filtered).T.reshape(-1) if len(filtered) > 1 else filtered[0]
    buf = array(typecode)
    buf.fromlist([int(v) for v in interleaved])
    return AudioSegment(data=buf.tobytes(), sample_width=seg.sample_width, frame_rate=sr, channels=seg.channels)


def _sox_silence(seg: AudioSegment, args: List[str]) -> AudioSegment:
    """SoX `silence -l 1 0 <threshold> -1 <max_keep> <threshold>`: cap any run of
    silence longer than <max_keep> seconds down to <max_keep> seconds."""
    max_ms = int(round(float(args[-2]) * 1000))
    thresh_dbfs = 20 * np.log10(float(str(args[-1]).rstrip("%")) / 100.0)
    parts = []
    prev_end = 0
    for start, end in detect_nonsilent(seg, min_silence_len=max_ms, silence_thresh=thresh_dbfs, seek_step=10):
        gap = start - prev_end
        parts.append(AudioSegment.silent(duration=max_ms, frame_rate=seg.frame_rate) if gap > max_ms else seg[prev_end:start])
        parts.append(seg[start:end])
        prev_end = end
    tail = len(seg) - prev_end
    parts.append(AudioSegment.silent(duration=max_ms, frame_rate=seg.frame_rate) if tail > max_ms else seg[prev_end:])
    out = seg[:0]
    for p in parts:
        out += p
    return out


def _sox_pad(seg: AudioSegment, args: List[str]) -> AudioSegment:
    """SoX `pad <secs>[@<position>]`: add silence, leading (position 0) by default."""
    lead_ms = trail_ms = 0
    for spec in args:
        length_str, _, position = str(spec).partition("@")
        ms = int(round(float(length_str) * 1000))
        if position in ("", "0"):
            lead_ms += ms
        else:
            trail_ms += ms
    out = seg
    if trail_ms:
        out = out + AudioSegment.silent(duration=trail_ms, frame_rate=seg.frame_rate)
    if lead_ms:
        out = AudioSegment.silent(duration=lead_ms, frame_rate=seg.frame_rate) + out
    return out


def apply_sox_chain(seg: AudioSegment, effects: List[list]) -> AudioSegment:
    """Drop-in replacement for torchaudio.sox_effects.apply_effects_tensor, limited
    to the effects actually configured in shared_module.py."""
    for effect in effects:
        if not effect:
            continue
        name, args = str(effect[0]), [str(a) for a in effect[1:]]
        if name == "norm":
            seg = _sox_norm(seg)
        elif name in ("highpass", "lowpass"):
            offset = 1 if args and args[0].startswith("-") else 0
            order = abs(int(args[0])) if offset else 1
            seg = _butter_filter(seg, name, float(args[offset]), order)
        elif name == "silence":
            seg = _sox_silence(seg, args)
        elif name == "pad":
            seg = _sox_pad(seg, args)
        else:
            red(f"Unsupported sox effect, skipping: {effect}")
    return seg


@optional_typecheck
def sound_preprocessing(audio_mp3_path: Union[PosixPath, str]) -> PosixPath:
    "removing silence, maybe try to enhance audio, apply filters etc"
    whi(f"Preprocessing {audio_mp3_path}")

    if audio_mp3_path is None:
        whi("Not cleaning sound because received None")
        return None

    assert "_proc" not in str(audio_mp3_path), f"Audio already processed apparently: {audio_mp3_path}"

    # load from file and apply the sox-equivalent chain in memory
    temp = AudioSegment.from_file(audio_mp3_path)
    temp = apply_sox_chain(temp, shared.preprocess_sox_effects)
    new_path = Path(audio_mp3_path).parent / (Path(audio_mp3_path).stem + "_proc" + Path(audio_mp3_path).suffix)
    temp.export(new_path, format="mp3")

    whi(f"Done preprocessing {audio_mp3_path} to {new_path}")
    return new_path

@optional_typecheck
def force_sound_processing(path: Optional[Union[str, PosixPath]] = None) -> PosixPath:
    """harsher sound processing for the currently loaded next audio. This is
    done if there are some residual long silence that are making whisper
    hallucinate. The previous audio will be moved to the dirload done_dir and
    the new processed sound will take its place"""
    assert shared.pv["enable_dirload"], "Incoherent UI"
    assert not shared.dirload_queue.empty, "Dirload queue is empty"

    expected_path = Path(shared.dirload_queue[shared.dirload_queue["loaded"] == True].iloc[0].name)
    if path is None:
        path = expected_path
    else:
        path = Path(path)
        if path != expected_path:
            red(f"Forced processing sound of {path} but expected probably {expected_path}")

    assert path.exists(), f"Missing {path}"
    red(f"Forcing harsher sound processing of {path}")

    out_path = shared.done_dir / (path.stem + "_unprocessed" + path.suffix)
    if out_path.exists():
        red(f"Output file already exists: {out_path}\nI will not replace it")

    # decode + filter fully in memory first, so a decode failure cannot leave the
    # file half-moved
    temp = AudioSegment.from_file(path)
    temp = apply_sox_chain(temp, shared.force_preprocess_sox_effects)

    try:
        red(f"Moving {path} to {out_path}")
        shutil.move(path, out_path)
        assert not path.exists(), f"{path} already exists"
        temp.export(path, format="mp3")
        gr.Warning(red(f"Done forced preprocessing {path}. The original is in {out_path}"))
        return path
    except Exception as err:
        gr.Warning(red(f"Error when processing sound: {err}"))
        # undo everything
        if out_path.exists():
            shutil.move(out_path, path)
        assert path.exists(), f"File was lost! {path}"
        return path



@optional_typecheck
def format_audio_component(
    audio: Union[str, gr.Audio, PosixPath, dict],
    ) -> Union[str, PosixPath]:
    """to make the whole UI faster and avoid sending multiple slightly
    differently processed audio to whisper: preprocessing and postprocessing
    are disabled but this sometimes make the audio component output a dict
    instead of the mp3 audio path. This fixes it while still keeping the cache
    working."""
    if isinstance(audio, dict):
        new_audio = audio["path"]
        if new_audio.startswith("http"):
            new_audio = new_audio.split("file=")[1]
        # whi(f"Preprocessed audio manually: '{audio}' -> '{new_audio}'")
        audio = new_audio
    elif isinstance(audio, (str, type(Path()))):
        # whi(f"No audio formating needed for '{audio}'")
        pass
    else:
        raise ValueError(red(f"Unexpected audio format for {audio}: {type(audio)}"))
    return audio


def rgb_to_bgr(image):
    """gradio is turning cv2's BGR colorspace into RGB, so
    I need to convert it again"""
    assert shared.pv["enable_gallery"], "Incoherent UI"
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


@optional_typecheck
def roll_queued_galleries(*qg: Optional[List[Union[List, Tuple]]]) -> List[Optional[Union[gr.Gallery, dict, List]]]:
    "pop the first queued gallery and send it to the main gallery"
    assert shared.pv["enable_queued_gallery"], "Incoherent UI"
    output = list(qg) + [None]

    # make sure to delete the queued gallery that are now None from profile
    for qg_cnt, qg in enumerate(range(1, shared.queued_gallery_slot_nb + 1)):
        img = output[qg_cnt]
        if img is None:
            getattr(shared.pv, f"save_queued_gallery_{qg:03d}")(None)

    return output


@optional_typecheck
def qg_add_to_new(*qg) -> List[Optional[Union[gr.Gallery, dict, List]]]:
    """triggered by a shortcut, will add from clipboard the image to
    a new queued gallery"""
    qg = list(qg)
    # find the index of the latest non empty gallery
    for i, img in enumerate(qg):
        if img is None:
            break
    i = max(0, min(i, shared.queued_gallery_slot_nb))
    new_img = get_image(qg[i])
    gr.Warning(red(f"Adding to new gallery #{i + 1}"))
    qg[i] = new_img
    return qg


@optional_typecheck
def qg_add_to_latest(*qg) -> List[Optional[Union[gr.Gallery, dict]]]:
    """triggered by a shortcut, will add from clipboard the image to
    the latest non empty queued gallery"""
    qg = list(qg)
    for i, img in enumerate(qg):
        if img is None:
            break
    i -= 1
    i = max(0, min(i, shared.queued_gallery_slot_nb))
    new_img = get_image(qg[i])
    qg[i] = new_img
    gr.Warning(red(f"Adding to latest gallery #{i + 1}"))
    return qg


@optional_typecheck
def create_audio_compo(**kwargs) -> gr.Microphone:
    defaults = {
            "type": "filepath",
            "format": "mp3",
            "value": None,
            "label": "Untitled",
            "show_label": True,
            "container": True,
            "buttons": ["download"],
            "elem_classes": ["js_audiocomponent"],
            "min_width": "100px",
            "waveform_options": {"show_recording_waveform": False},
            "editable": True,
            "scale": 1,
            }
    defaults.update(kwargs)
    return gr.Microphone(**defaults)


@optional_typecheck
def roll_audio(*slots) -> List[Optional[Union[dict, str]]]:
    assert len(slots) > 1, f"invalid number of audio slots: {len(slots)}"
    assert isinstance(slots, tuple), f"unexpected slots type: {slots}"
    slots = list(slots)

    # gradio >=6 hands recorded audio over as FileData dicts, which are not
    # valid output values: convert them to update dicts before any early
    # return so they can be sent back to the audio components
    for i, s in enumerate(slots):
        if isinstance(s, dict) and s.get("__type__") != "update":
            slots[i] = {
                    "__type__": "update",  # this is how gr.update works
                    "label": s.get("orig_name", "New"),
                    "value": s.get("path"),
                    }

    if all((slot is None for slot in slots)):
        return slots
    if all((slot is None for slot in slots[1:])):
        return slots

    slots.pop(0)

    # update the name of each audio to its neighbour
    for i, s in enumerate(slots):
        if s is None:
            continue
        elif isinstance(s, str):
            # it's already a path, no need to modify it
            continue
        elif isinstance(s, dict):
            if s.get("__type__") == "update":
                # already converted to an update dict above
                continue
            slots[i] = {
                    "__type__": "update",  # this is how gr.update works
                    "label": s.get("orig_name", "New"),
                    "value": s.get("path"),
                    }
    while None in slots:
        slots.remove(None)

    while len(slots) < shared.audio_slot_nb:
        slots.append(
                {
                    "__type__": "update",
                    "label": "New",
                    "value": None,
                    }
                )

    return slots


head = """
<style>
mark {
    background-color: #5767AE;
    color: white !important;
}
.dark mark {
    background-color: #5767AE;
    color: white;
}

.separator {
    height: 1px;
    background-color: #5767AE;
    margin-top: 0;
    margin-bottom: 0;
}
.dark .separator {
    height: 1px;
    background-color: #5767AE;
    margin-top: 0;
    margin-bottom: 0;
}

.scrollablecontent {
    min-height: 220px !important;
    max-height: 220px !important;
    overflow-y: auto;
}
</style>
<div class="scrollablecontent">
"""
# min-height: 5em !important;
# max-height: 5em !important;
# line-height: 1em;
div_separator = '  <div class="separator">-</div>  '
tail = """
</div>
"""

@optional_typecheck
def update_audio_slots_txts(gui_enable_dirload: bool, *audio_slots_txts) -> List[Optional[str]]:
    """ran frequently to update the content of the textbox of each pending
    audio to display the transcription and cloze
    """
    if gui_enable_dirload is False:
        # the components are invisible so return None
        return [None for i in audio_slots_txts]

    df = shared.dirload_queue
    if df.empty:
        return [f"{head}<mark>Dirload not yet loaded</mark>{tail}" for i in audio_slots_txts]

    try:
        df = df[df["loaded"] == True]
        if df.empty:
            return [f"{head}<mark>Empty</mark>{tail}" for i in audio_slots_txts]

        trans = [t.strip() if isinstance(t, str) else t for t in df["transcribed"].tolist()]
        while len(trans) < len(audio_slots_txts):
            trans.append("Pending?")

        alf = [a.strip() if isinstance(a, str) else a for a in df["alfreded"].tolist()]
        while len(alf) < len(trans):
            alf.append("Pending?")

        output = [f"{t}{div_separator}{f}" for t, f in zip(trans, alf)]

        for i, o in enumerate(output):
            o = re.sub("alfred", "<mark>alfred</mark>", o, flags=re.IGNORECASE)
            o = re.sub(" fred", " <mark>alfred</mark>", o, flags=re.IGNORECASE)
            o = re.sub("carte", "<mark>carte</mark>", o, flags=re.IGNORECASE)
            o = re.sub("note", "<mark>note</mark>", o, flags=re.IGNORECASE)
            o = re.sub("<thoughts>", "<mark>thoughts</mark>", o, flags=re.IGNORECASE)
            o = re.sub("</thoughts>", "<mark>thoughts</mark>", o, flags=re.IGNORECASE)

            o = re.sub(r"\Wstop ?\W", "<mark>stop</mark>", o, flags=re.IGNORECASE)
            o = re.sub(r"\W#####\W", "<mark>#####</mark>", o, flags=re.IGNORECASE)

            o = re.sub("started", "<mark>started</mark>", o, flags=re.IGNORECASE)
            o = re.sub("Pending?", "<mark>Pending?</mark>", o)
            output[i] = head + o + tail

        return output
    except Exception as err:
        return [f"{head}<mark>{err}</mark>{tail}" for i in audio_slots_txts]
