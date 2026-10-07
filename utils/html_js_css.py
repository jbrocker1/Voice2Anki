from .shared_module import shared

# Persistent user-chosen accent color. Loaded once at module import so
# both js_load (page-load paint) and the CSS override block below can
# see it. Empty string = use the Soft theme's default orange.
try:
    accent = shared.pv["accent_color"]
except (KeyError, TypeError):
    accent = ""

darkmode_js = """() => {
    if (document.querySelectorAll('.dark').length) {
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    } else {
        document.querySelector('body').classList.add('dark');
    }
}"""

html_head = """
<script>

function unfocus(e) {
    document.activeElement.blur();
    document.body.focus();
    document.documentElement.focus();
    e.preventDefault();  // make sure to avoid scrolling after pressing spacebar
}

function shortcuts(e) {
    // the switch is so that keypress are ignored if an input element is
    // in focus
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {

        // unselect anything by pressing shift+space or escape
        case "input":
        case "textarea":
            if ((e.code == 'Space' && e.shiftKey) || (e.key == 'Escape') || (e.keyCode == 27)) {
                unfocus(e);
            }

        //case "select":
        //case "button":
        break;

        default:
        // suspend previous card
        if (e.code == "KeyS" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_suspendpreviousbtn").click();
        }
        // mark previous card
        else if (e.code == 'Semicolon' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_markpreviousbtn").click();
        }
        // untoggle check next card
        else if (e.key == 'm') {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_marknext").children[1].children[0].checked = false;
        }
        // get card status
        else if (e.key == "s") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_cardstatusbtn").click();
        }
        // recur improvement
        else if (e.code == "KeyF" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_llmfeedbackbtn").click();
        }

        // toggle nightmode
        else if (e.code == "KeyN" && e.shiftKey) {
            document.getElementById("js_darkmodebtn").click();
        }

        // select textbox
        else if (e.key == "e") {
            e.preventDefault();  // dont type the e
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            ch = document.getElementById("js_txtchatgpt").children;
            ch[ch.length - 1].focus();
        }
        else if (e.code == "KeyE" && e.shiftKey) {
            e.preventDefault();  // dont type the e
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            ch = document.getElementById("js_txtwhisper").children;
            ch[ch.length - 1].focus();
        }

        // roll 1 2 3
        else if (e.key == "&") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll1").click();
        }
        else if (e.key == "é") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll12").click();
        }
        else if (e.key == '"') {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll123").click();
        }

        // 123
        else if (e.key == "3" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_toankibtn").click();
        }
        else if (e.key == "2" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_transcriptbtn").click();
        }
        else if (e.key == "1" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_transcribebtn").click();
        }

        // roll gallery
        else if (document.getElementById('js_guienablequeuedgallery').children[1].children[0].checked == true && e.code == 'KeyG' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            if (confirm("Roll gallery?")) {
                document.getElementById("js_rollgallbtn").click();
            }
        }
        // add to next queued gallery
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyR' && e.shiftKey) {
            if (!document.querySelector('.js_tabqueues').checkVisibility()) { alert("Shortcut only available in tab 'Queues'") ; return ;}
            // only active if the right tabs are enabled
            if (document.querySelector('.js_tabqueues').checkVisibility() && document.querySelector('.js_queueqgclass').checkVisibility()) {
                    document.getElementById("js_btnqgnew").click();
            }
            else {
                alert("To add to queued gallery, go to 'Queues' then 'Queud audio'");
            }
        }
        // append to latest queueud gallery
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyQ' && e.shiftKey) {
            if (!document.querySelector('.js_tabqueues').checkVisibility()) { alert("Shortcut only available in tab 'Queues'") ; return ;}
            // only active if the right tabs are enabled
            if (document.querySelector('.js_tabqueues').checkVisibility() && document.querySelector('.js_queueqgclass').checkVisibility()) {
                document.getElementById("js_btnqgadd").click();
            }
            else {
                alert("To add to queued gallery, go to 'Queues' then 'Queud audio'");
            }
        }

        // dirload
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyD' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            if (confirm("Load from dir?")) {
                document.getElementById("js_dirloadbtn").click();
            }
        }

        // switch top tabs
        else if (e.code == "KeyT") {
            var tabs = document.querySelectorAll(".js_toptabs")[0].parentElement.childNodes[0].children;
            Array.from(tabs).forEach((tab, index) => {
                if (tab.ariaSelected == 'true') {
                    if (event.shiftKey) {
                        var newIndex = (index > 0) ? index - 1 : tabs.length - 1;
                    } else {
                        var newIndex = (index + 1) % tabs.length;
                    }
                    tabs[newIndex].click();
                    return;
                }
            });
        }

        // unfocus
        else if ((e.code == 'Space' && e.shiftKey) || (e.key == 'Escape') || (e.keyCode == 27)) {
            unfocus(e);
        }

        // ignore space
        else if ((e.code == 'Space' && e.shiftKey) || (e.code == 'Space')) {
            // do nothing
        }

        // no shortcut found
        else {
            alert(`Unrecognized shortcut: ${e.key} (or ${e.code})`);
            }

        }
}
function tabswitcher(e) {
    // tab to switch tab
    if (event.key === 'Tab') {
        event.preventDefault();

        // the subtabs cycled depend on the main tab focused
        if (document.querySelector('.js_tabmain').checkVisibility()) {
            var selector = '.js_subtab_main'
        }
        else if (document.querySelector('.js_tabsettings').checkVisibility()) {
            var selector = '.js_subtab_settings'
        }
        else if (document.querySelector('.js_tabqueues').checkVisibility()) {
            var selector = '.js_subtab_queues'
        }
        else if (document.querySelector('.js_tabmemoriesandbuffer').checkVisibility()) {
            var selector = '.js_subtab_memoriesandbuffer'
        }
        else {
            // alert("No subtab to switch here.");
            return;
        }


        var tabs = document.querySelectorAll(selector)[0].parentElement.childNodes[0].children;
        Array.from(tabs).forEach((tab, index) => {
            if (tab.ariaSelected == 'true') {
                if (event.shiftKey) {
                    var newIndex = (index > 0) ? index - 1 : tabs.length - 1;
                } else {
                    var newIndex = (index + 1) % tabs.length;
                }
                tabs[newIndex].click();
                return;
            }
        });

    }
}

document.addEventListener('keypress', shortcuts, false);
document.addEventListener('keydown', tabswitcher, false);


////// code related to syntax highlighting
//const rules = [
//  { regex: /\b(for)\b/g, replacement: '<span style="color: red;">$1</span>' },
//  { regex: /\b(if)\b/g, replacement: '<span style="color: blue;">$1</span>' },
//  { regex: /\b(else)\b/g, replacement: '<span style="color: green;">$1</span>' },
//];
//// Apply syntax highlighting
//function applySyntaxHighlighting(html) {
//  rules.forEach(rule => {
//    html = html.replace(rule.regex, rule.replacement);
//  });
//  return html;
//}
//// Function to preserve caret position
//function getCaretPosition(editableDiv) {
//  let caretPos = 0, sel, range;
//  if (window.getSelection) {
//    sel = window.getSelection();
//    if (sel.rangeCount) {
//      range = sel.getRangeAt(0);
//      if (range.commonAncestorContainer === editableDiv.parentElement) {
//        caretPos = range.endOffset;
//      }
//    }
//  }
//  return caretPos;
//}
//// Set caret position
//function setCaretPosition(editableDiv, position) {
//  if (window.getSelection && document.createRange) {
//    const range = document.createRange();
//    range.selectNodeContents(editableDiv.parentElement);
//    range.collapse(true);
//    range.setStart(editableDiv.parentElement, position);
//    range.setEnd(editableDiv.parentElement, position);
//    const sel = window.getSelection();
//    sel.removeAllRanges();
//    sel.addRange(range);
//  }
//}
//// Main function to handle input and styling
//function handleInput(event) {
//  const target = event.target;
//  const caretPosition = getCaretPosition(target);
//  let content = target.innerText;
//  target.innerHTML = applySyntaxHighlighting(content);
//  setCaretPosition(target, caretPosition);
//}
//
//el=document.getElementById("js_txtchatgpt").childNodes[1].childNodes[5];
//el.contentEditable = true;
//el.addEventListener('input', handleInput);


//// code to create a text input area on top of txtchatgpt:
//// Assuming you have an existing element with id 'existingElement'
//const existingElement = document.getElementById('js_txtchatgpt');
//
//// Create new text input element
//const newTextElement = document.createElement('input');
//newTextElement.type = 'text';
//
//// Get coordinates and dimensions of existingElement
//const rect = existingElement.childNodes[1].childNodes[5].getBoundingClientRect();
//
//// Set style of newTextElement for exact overlay based on viewport position
//newTextElement.style.position = 'absolute';
//newTextElement.style.top = `${rect.top + window.scrollY}px`; // Adjust for scrolling
//newTextElement.style.left = `${rect.left + window.scrollX}px`; // Adjust for scrolling
//newTextElement.style.width = `${rect.width}px`;
//newTextElement.style.height = `${rect.height}px`;
//
//// Link contents
//newTextElement.oninput = () => existingElement.childNodes[1].childNodes[5].value = newTextElement.value;
////existingElement.oninput = () => newTextElement.value = existingElement.childNodes[1].childNodes[5].value;
//
//// Append newTextElement to body to ensure it is positioned based on viewport coordinates
//document.body.appendChild(newTextElement);


</script>
"""

# dynamically adjust the height of the app to avoid scrolling up abruptly
js_longer = """() => {
    document.querySelectorAll(".app")[0].style.height='5000px';
}
"""
js_reset_height = """() => {
    document.querySelectorAll(".app")[0].style.height='';
}
"""

# executed on load
js_load = """() => {
    // make sure the audios keep the same size even when they are unset.
    // Height was 2.3x component height + 90px min, which pushed the
    // Transcribe / Clozify / Ankify actions below the fold for an empty
    // workspace; 1.2x + 70px keeps the recorder visible while letting
    // the action row sit above the fold.
    var h = Math.max(70, Math.floor(1.2 * document.getElementsByClassName("js_audiocomponent")[0].clientHeight));


    Array.from(document.getElementsByClassName("js_audiocomponent")).forEach(el => el.style.height = `${h}px`)

}
"""
# NOTE: an earlier version of this script also painted the persisted accent
# color on every primary button at mount. It hung the gradio bundle on load
# (90 s wait_for_function timeout, page stuck on the splash) and was reverted.
# The persisted color still works -- the ColorPicker in Settings writes to
# shared.pv["accent_color"], which ValueStorage pickles to
# profiles/<name>/accent_color.pickle, and the CSS-variable override in
# the ``css`` string below paints a subset of consumers. Gradle 6's
# Svelte-scoped primary-button styles win every CSS-variable / theme-token /
# ``!important`` cascade, so the visible button paint stays orange despite
# the persisted accent. Revisit with a DOM-walking JS paint if the visual
# override matters in practice.

css = """
/* make sure those tabs take all the width */
#js_widetabs-button { flex-grow: 1 !important;}
"""

if shared.big_font:
    css += """
/* Larger font for some text elements */
#js_txtchatgpt > label > textarea {font-size: 20px;}
#js_txtwhisper > label > textarea {font-size: 20px;}
"""
# else:
#     css += """
# /* Larger font for some text elements */
# #js_txtchatgpt > label > textarea {font-size: 17px;}
# #js_txtwhisper > label > textarea {font-size: 17px;}
# """

if shared.widen_screen:
    css += "\n.app { max-width: 100% !important; }"
    css += "\n.app { width: 100% !important; }"

# Persistent user-chosen accent color. ValueStorage pickles the value
# to ``profiles/<profile>/accent_color.pickle`` whenever the user changes
# it via the Settings ColorPicker, so it survives ``shared.reset()`` and
# a full app restart. ``accent`` is loaded at module top so the js_load
# string below can interpolate it. Empty string = use the Soft theme's
# default orange.
if accent and isinstance(accent, str) and accent.startswith("#") and len(accent) in (7, 9):
    # Compute a darker hover variant. Use colorsys if available, fall back
    # to a fixed darkening shortcut otherwise.
    try:
        import colorsys
        h_hex = accent.lstrip("#")
        r = int(h_hex[0:2], 16) / 255
        g = int(h_hex[2:4], 16) / 255
        b = int(h_hex[4:6], 16) / 255
        hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
        ll_dark = max(0.0, ll - 0.1)
        rr, gg, bb = colorsys.hls_to_rgb(hh, ll_dark, ss)
        accent_hover = f"#{int(rr*255):02x}{int(gg*255):02x}{int(bb*255):02x}"
    except Exception:
        accent_hover = accent
    css += f"""
/* Persistent user accent color -- the CSS-variable overrides that the
   picker would otherwise use. They only paint a subset of components
   (those that actually read ``var(--button-primary-*)``); the rest of
   the primary button paint is done by an inline-style DOM walker set
   up by the ColorPicker (see utils/gui.py accent_color_picker.change).
   We keep the CSS overrides anyway: cheaper, work for the cases they
   cover, and serve as a safety net if the JS path ever regresses. */
:root {{
    --button-primary-background-fill: {accent} !important;
    --button-primary-background-fill-hover: {accent_hover} !important;
    --button-primary-border-color: {accent} !important;
    --button-primary-border-color-hover: {accent_hover} !important;
}}
"""

