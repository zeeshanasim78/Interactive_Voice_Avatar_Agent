================================================================================
NAB AI - INTERACTIVE VOICE AVATAR AGENT
README - SETUP, BUILD, TEST & DEPLOYMENT GUIDE (Python 3.12 Edition)
================================================================================

This file explains, step by step, how to actually BUILD the system described in
"NAB_AI_System_Design.md" using VS Code, Git, and the Claude Code CLI, how to
test it, and how to deploy it to another machine. Follow the sections in order.

This edition targets Python 3.12 specifically. Every library named below was
checked against the current Python 3.12 ecosystem, and a few package names
differ from what older tutorials online might show you (e.g. coqui-tts instead
of TTS, PySide6 instead of PyQt5, silero-vad instead of webrtcvad,
langchain-text-splitters instead of the full langchain package). See Section
15 ("Python 3.12 Migration - Library Changes & Rationale") and Section 16
("Second Independent Verification Pass") in NAB_AI_System_Design.md for the
full reasoning behind each swap and confirmation that no requirement is lost.

Keep NAB_AI_System_Design.md open next to this file - every step below refers
back to a specific Phase (0-10) and Section number in that document.

--------------------------------------------------------------------------------
0. WHAT YOU NEED BEFORE YOU START
--------------------------------------------------------------------------------
- A Windows or Linux PC (8 GB RAM minimum, 16 GB+ recommended - see Section 10
  "Hardware Guidance" in the design doc).
- Internet connection (only needed for the ONE-TIME setup/download step; the
  finished system runs fully offline).
- A free GitHub account (for Git/version control).
- Node.js is NOT required. **Python 3.12** IS required (this project's code
  and libraries are all verified against Python 3.12 - see Section 15/16 of
  the design doc for exactly which libraries were swapped to support it and
  why).
- A Claude.ai / Anthropic account with Claude Code CLI access (or another
  coding assistant / write the code manually using this README + the MD file).

--------------------------------------------------------------------------------
1. INSTALL CORE SOFTWARE
--------------------------------------------------------------------------------
1.1 Install Git
    Windows : download from https://git-scm.com/downloads and run the installer
              (accept defaults).
    Linux   : sudo apt update && sudo apt install git -y

    Verify:
        git --version

1.2 Install VS Code
    Download from https://code.visualstudio.com and install it.
    Inside VS Code, install these extensions (Extensions panel, Ctrl+Shift+X):
        - "Python" (by Microsoft)
        - "Pylance"
        - "GitLens" (optional, helpful for Git history)

1.3 Install Python 3.12 (do not use 3.13 yet unless you've re-verified the
    library table in Section 4/15 of the design doc against it; do not use
    3.10/3.11 - several libraries in this stack specifically require the
    Python-3.12-compatible package names/versions listed there)
    Windows : download the Python 3.12.x installer from
              https://www.python.org/downloads/ - during install, tick
              "Add Python to PATH".
    Linux   : sudo apt install python3.12 python3.12-venv python3-pip -y

    Verify:
        python --version        (Windows - should print "Python 3.12.x")
        python3.12 --version    (Linux)

1.4 Install Ollama (runs the local, free LLM used for RAG answers)
    Download from https://ollama.com and install it.
    Then open a terminal and pull a model (one-time download, ~4-5 GB):
        ollama pull llama3.1:8b

    Verify Ollama is running:
        ollama list

1.5 Install Claude Code CLI
    Follow the official install instructions at:
        https://docs.claude.com/en/docs/claude-code
    (the exact install command can change, so check that page - typically it
    is installed via npm or a shell script provided there).
    After installing, sign in:
        claude login
    Verify:
        claude --version

--------------------------------------------------------------------------------
2. CREATE THE PROJECT AND SET UP GIT VERSION CONTROL
--------------------------------------------------------------------------------
2.1 Create the project folder and initialize Git
        mkdir nab_ai
        cd nab_ai
        git init

2.2 Create a .gitignore file (so venv, models, and large media are not committed)
    Create a file named ".gitignore" in the nab_ai folder with these lines:

        venv/
        __pycache__/
        *.pyc
        knowledge_base/vector_store/
        logs/*.db
        *.wav
        *.mp4
        .env

2.3 Create a GitHub repository
    - Go to https://github.com and click "New repository".
    - Name it "nab-ai", keep it Private (recommended, since it relates to a
      government agency's public-facing tool), do not initialize with a README
      (you already have one).
    - Copy the remote URL GitHub shows you (e.g. git@github.com:yourname/nab-ai.git
      or https://github.com/yourname/nab-ai.git).

2.4 Connect your local folder to GitHub
        git remote add origin <PASTE_YOUR_REPO_URL_HERE>
        git branch -M main

2.5 Copy the two files you already have into this folder:
    - NAB_AI_System_Design.md
    - README.txt (this file)

2.6 Make your first commit
        git add .
        git commit -m "Initial commit: design doc and README"
        git push -u origin main

From now on, after finishing each Phase in Section 7 of the design doc, commit
your work:
        git add .
        git commit -m "Phase 2: speech-to-text module working"
        git push

--------------------------------------------------------------------------------
3. OPEN THE PROJECT IN VS CODE AND CREATE A VIRTUAL ENVIRONMENT
--------------------------------------------------------------------------------
3.1 Open VS Code, then File > Open Folder > select the "nab_ai" folder.

3.2 Open a terminal inside VS Code (Terminal > New Terminal) and create a
    virtual environment:
        Windows : python -m venv venv
                Note : You can also select using the Ctrl + Shift + P and Create Python environment
                
                To activate Virtual environment use command
                venv\Scripts\activate

                Note : In case you are getting error while activating the Virtual environment
                        Execute the following command on Power Shell via Administrator

                        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

                        Restart VS Code and try opening a terminal again

        Linux   : python3.12 -m venv venv
                  source venv/bin/activate

    You should see "(venv)" at the start of your terminal prompt once active.
    In VS Code, click the Python version in the bottom-right status bar and
    select the "venv" interpreter so VS Code uses it too. Confirm it says
    3.12.x, not an older version, if more than one Python is installed.

3.3 Create the folder structure exactly as described in Section 6 of the
    design doc ("Project Folder Structure"). In the VS Code terminal:

        Note : YOu can execute these comman on the CMD Terminal while residing on the project start folder

        mkdir config knowledge_base knowledge_base\pdfs knowledge_base\vector_store
        mkdir videos avatar_assets avatar_assets\audio logs
        mkdir src src\audio src\routing src\video src\rag src\response src\ui src\admin

    (On Linux, use forward slashes and "mkdir -p" instead, e.g.
     mkdir -p knowledge_base/pdfs knowledge_base/vector_store)

3.4 Create requirements.txt in the project root with (these are the
    Python-3.12-verified packages from Section 4/15 of the design doc -
    note several package names differ from what you might see in older
    tutorials, on purpose):

        sounddevice
        silero-vad
        faster-whisper
        rapidfuzz
        sentence-transformers
        python-vlc
        pdfplumber
        chromadb
        langchain-text-splitters
        pyyaml
        coqui-tts
        pyttsx3
        PySide6
        argostranslate>=1.9.4

3.5 Install everything:
        pip install -r requirements.txt

    NOTE: "coqui-tts" and "sentence-transformers" are large downloads and can
    take a while the first time. This is expected.

    IMPORTANT: do NOT run "pip install TTS" (no dash, capital letters) or
    "pip install PyQt5" or "pip install webrtcvad" or plain "pip install
    langchain" or "pip install argos-translate" (with a hyphen - this
    package does not exist on PyPI and will fail with "No matching
    distribution found") - those are the OLD/WRONG package names that
    either fail to install or fail to import correctly on Python 3.12. Use
    exactly the names above (note "argostranslate" has NO hyphen).
    See Section 15 of NAB_AI_System_Design.md for the full explanation of
    each swap, and Section 15.1 for this specific correction.

--------------------------------------------------------------------------------
4. USE CLAUDE CODE CLI TO BUILD EACH PHASE (SPEC-DRIVEN, BEGINNER-FRIENDLY)
--------------------------------------------------------------------------------
The idea: you do NOT hand Claude Code the whole project at once. You feed it
ONE phase from Section 7 of NAB_AI_System_Design.md at a time, let it write
that module, test it, commit it, then move to the next phase. This keeps every
step small, working, and reviewable - exactly what "spec-driven development"
means.

4.1 From the project root (with venv activated), start Claude Code:
        claude

4.2 For each phase, use a prompt of this shape (type it into the Claude Code
    session):
        Phase - I - Prompt
        "Read NAB_AI_System_Design.md. This project targets Python 3.12 - use
        exactly the libraries named in Section 4/15 of that document (e.g.
        silero-vad, not webrtcvad; coqui-tts, not TTS; PySide6, not PyQt5;
        langchain-text-splitters, not full langchain). Implement Phase 1
        exactly as specified in Section 7 (Voice Capture & Wake Word) and
        Section 5.1. Create src/audio/capture.py and src/audio/wake_word.py.
        Follow the acceptance criteria listed for this phase. Do not implement
        any later phase yet."

        Phase - 2 - Prompt
        "Read NAB_AI_System_Design.md. This project targets Python 3.12 - use
        exactly the libraries named in Section 4/15 of that document (e.g.
        silero-vad, not webrtcvad; coqui-tts, not TTS; PySide6, not PyQt5;
        langchain-text-splitters, not full langchain). Implement Phase 2
        exactly as specified in Section 7 (Speech-to-Text (STT)) and
        Section 5.2. Create src/audio/stt.py file.
        Follow the acceptance criteria listed for this phase. Do not implement
        any later phase yet."

        Phase - 3 - Prompt
        "Read NAB_AI_System_Design.md. This project targets Python 3.12 - use
        exactly the libraries named in Section 4/15 of that document (e.g.
        silero-vad, not webrtcvad; coqui-tts, not TTS; PySide6, not PyQt5;
        langchain-text-splitters, not full langchain). Implement Phase 3
        exactly as specified in Section 7 (Intent Recognition & Routing Engine) and
        Section 5.3. Create following file : config/restricted_topics.yaml, 
        src/routing/restricted_filter.py and src/routing/intent_router.py
        Follow the acceptance criteria listed for this phase. Do not implement
        any later phase yet."


    Reusing this "targets Python 3.12, use exactly the libraries named in
    Section 4/15" sentence at the start of every phase prompt (Phases 1-9)
    is important - it stops Claude Code (or any assistant) from defaulting
    to the more commonly-known older package names from its training data.

4.3 Repeat this pattern phase by phase, in order:
        Phase 0  -> environment/dependency sanity check (already mostly done above)
        Phase 1  -> src/audio/capture.py, src/audio/wake_word.py
                    (NOTE: silero-vad's API is different from the old
                    webrtcvad's - it works on a stream/tensor and returns
                    speech-segment timestamps, not a simple per-frame
                    is_speech() call. Tell Claude Code to write directly
                    against silero-vad's real API, not adapt old webrtcvad
                    code - see Section 16.2 of the design doc.)
        Phase 2  -> src/audio/stt.py
        Phase 3  -> config/restricted_topics.yaml, src/routing/restricted_filter.py,
                    src/routing/intent_router.py
        Phase 4  -> config/video_map.yaml, src/video/player.py
        Phase 5  -> src/rag/ingest.py, src/rag/query_engine.py
        Phase 6  -> config/settings.yaml (fallback + processing-prompt text)
        Phase 7  -> src/response/tts.py, src/response/avatar.py,
                    avatar_assets/audio/processing_prompt.wav (generated once)
                    (NOTE: coqui-tts keeps the same "from TTS.api import TTS"
                    import even though the pip package is named coqui-tts -
                    this is expected, not a typo.)
        Phase 8  -> src/orchestrator.py, logs/interactions.db wiring
        Phase 9  -> src/ui/kiosk_app.py
                    (NOTE: use PySide6 imports, e.g. "from PySide6.QtWidgets
                    import ..." and "app.exec()" not "app.exec_()" - PySide6
                    is ~99% like PyQt5 but these two details differ.)
        Phase 10 -> end-to-end test scripts (see Section 5 below)

4.4 After Claude Code finishes writing a phase's files, ALWAYS:
        a) Read the code it produced (don't skip this).
        b) Run the phase's own test from Section 7 of the design doc.
        c) Only once the acceptance criteria pass, commit:
                git add .
                git commit -m "Phase X: <short description>"
                git push

4.5 If a phase's test fails, tell Claude Code exactly what went wrong (paste
    the error message) and ask it to fix only that phase - do not move ahead
    with a broken phase.

--------------------------------------------------------------------------------
5. RUNNING TESTS FOR EACH PHASE AND END-TO-END
--------------------------------------------------------------------------------
5.1 Install a test runner:
        pip install pytest

5.2 Ask Claude Code to generate a small test file per phase, e.g. for Phase 2:

        "Create tests/test_stt.py using pytest. It should load 3 short sample
        WAV files from tests/samples/ and assert the transcribed text roughly
        matches the expected text (use a simple word-overlap check, not exact
        match)."

    Run it with:
        pytest tests/test_stt.py -v

5.3 Manual acceptance checks (per Section 7 and Section 8 of the design doc):
    - Phase 3/5 routing test: create a CSV/spreadsheet of 30-60 sample phrases
      (English + Urdu) with their expected category (video / RAG / restricted /
      fallback). Ask Claude Code to write a script that runs each phrase
      through src/routing/intent_router.py and prints pass/fail against the
      expected category. Target: 90%+ accuracy, 100% on restricted-topic
      phrases (see Section 8, table row FR12).
    - Phase 7 processing-prompt test (FR14): trigger avatar.play_processing()
      directly and time how long it takes for audio to start. Target: under
      300 ms, every time (see Section 8 table row "FR14").
    - Phase 8 full pipeline test: run 10 full interactions covering all 4
      response types (video, RAG answer, restricted, fallback) and open
      logs/interactions.db (e.g. with the free "DB Browser for SQLite" tool)
      to confirm every interaction was logged with a complete row.

5.4 End-to-end test with real people (Phase 10):
    - Have 3-5 people who did NOT build the system try it live, in both
      English and Urdu, using only voice (no hints).
    - Track: was every response correct? Did the "please wait, processing"
      message play every single time there was a delay? Did anything crash?
    - Log issues as GitHub Issues in your repo so they're tracked, e.g.:
        (in your browser, on your GitHub repo page) Issues > New Issue

--------------------------------------------------------------------------------
6. RUNNING THE SYSTEM LOCALLY
--------------------------------------------------------------------------------
6.1 Make sure Ollama is running in the background (it usually auto-starts
    after install; otherwise run "ollama serve" in a terminal).

6.2 From the project root, with venv activated:
        python main.py

6.3 The kiosk window (Section 5.10 / Phase 9) should open full-screen showing
    the avatar in its idle state, ready to listen.

--------------------------------------------------------------------------------
7. DEPLOYING THE SYSTEM TO ANOTHER MACHINE
--------------------------------------------------------------------------------
Option A - Copy the whole project (simplest, recommended for a single kiosk PC)

7.1 On the new machine, repeat Section 1 (install Git, **Python 3.12**, Ollama)
    and pull a model with "ollama pull llama3.1:8b" (this re-downloads the
    model locally on that machine - it is not something you can just copy as
    a file easily across different OS/hardware, so re-pulling is simplest).

7.2 Clone your repository:
        git clone <YOUR_REPO_URL>
        cd nab_ai

7.3 Copy over (via USB drive or secure file transfer, NOT via Git, since these
    are excluded by .gitignore and can be large):
        - the /videos folder (your actual NAB video files)
        - the /knowledge_base/pdfs folder (your actual NAB PDF documents)
        - the /avatar_assets folder if it contains large media

7.4 Set up the environment exactly as in Section 3:
        python -m venv venv
        (activate it)
        pip install -r requirements.txt

7.5 Re-run the one-time PDF ingestion step (Phase 5) so the vector database
    is rebuilt fresh on this machine:
        python src/rag/ingest.py

7.6 Re-generate the cached processing-prompt audio (Phase 7, Section 5.8 of
    the design doc) if it wasn't copied over:
        python -c "from src.response.tts import generate_processing_prompt; generate_processing_prompt()"
    (adjust to whatever function name Claude Code actually generated)

7.7 Run it:
        python main.py

7.8 Make it auto-start (Section 12, "Deployment & Maintenance Notes" in the
    design doc):
        Linux (systemd) :
            Create /etc/systemd/system/nabai.service with a [Service] block
            that runs "python /path/to/nab_ai/main.py" as ExecStart, then:
                sudo systemctl enable nabai
                sudo systemctl start nabai
        Windows :
            Use Task Scheduler > Create Task > Trigger "At log on" > Action
            "Start a program" pointing to your venv's python.exe and main.py.

Option B - Package as a ZIP for a non-technical colleague to run
7.9 On your dev machine:
        git archive -o nab_ai_release.zip HEAD
    Then manually add the /videos, /knowledge_base/pdfs, and any large
    /avatar_assets files into that ZIP (since Git excludes them), and share
    the ZIP + this README with instructions to follow Sections 1, 3, 6, 7
    above on the target machine.

--------------------------------------------------------------------------------
8. IF YOUR OWN MACHINE'S HARDWARE ISN'T ENOUGH: USING GOOGLE COLAB
--------------------------------------------------------------------------------
Google Colab (https://colab.research.google.com) gives you a free, temporary
GPU-backed Linux notebook environment. It is genuinely useful here, but ONLY
for certain parts of this project - not for the whole finished kiosk app.

WHAT COLAB IS GOOD FOR:
- Prototyping and load-testing the heavy AI parts before you commit to
  hardware: Whisper transcription speed, the RAG pipeline (embeddings +
  Chroma + a local LLM), and Coqui TTS voice quality - all without needing a
  powerful PC yet.
- Running Phase 2 (STT), Phase 5 (RAG ingestion/query), and Phase 7 (TTS)
  in isolation, using uploaded sample audio/PDF files, to validate accuracy
  and timing before building the full kiosk app.
- Quickly comparing model sizes (e.g., Whisper "small" vs "medium", or
  Llama 3.1 8B vs a smaller quantized model) using Colab's free GPU, so you
  know what your real kiosk PC needs to buy/have (see Section 10 of the
  design doc, "Hardware Guidance").

WHAT COLAB IS NOT GOOD FOR:
- The finished, deployed kiosk app itself. Colab notebooks cannot reliably
  access a physical microphone/speaker/full-screen avatar window running on
  the museum/office kiosk PC, and a free Colab session disconnects after
  a period of inactivity or a few hours - unsuitable for a 24/7 public kiosk.
- Ollama typically is not the right fit inside Colab; for LLM testing in
  Colab, you would instead load a Hugging Face model directly via the
  "transformers" library within the notebook, purely to validate quality
  and speed, then still run the real deployed kiosk using Ollama locally.

HOW TO USE COLAB FOR PROTOTYPING (quick steps):
8.1 Go to https://colab.research.google.com and create a New Notebook.
8.2 Runtime > Change runtime type > Hardware accelerator > GPU (free tier).
8.3 In a cell, install the same libraries you need to test, e.g.:
        !pip install faster-whisper sentence-transformers chromadb pdfplumber coqui-tts
    (Colab's default runtime is usually already on Python 3.11/3.12 - either
    way, use "coqui-tts", not the old "TTS" package name, for the same
    Python-3.12-compatibility reasons explained in Section 15 of the design
    doc.)
8.4 Upload a few sample PDF/audio files using the folder icon on the left
    sidebar (or mount Google Drive with "from google.colab import drive").
8.5 Copy in just the relevant module code (e.g., src/rag/ingest.py and
    src/rag/query_engine.py) and run it against your sample PDFs to see how
    fast/accurate retrieval and generation are on Colab's GPU.
8.6 Use these results to decide your real kiosk PC's spec (Section 10 of the
    design doc), then build and test the actual system locally as described
    in Sections 1-7 of this README.

--------------------------------------------------------------------------------
9. QUICK COMMAND CHEAT SHEET
--------------------------------------------------------------------------------
Git:
    git status                 - see what's changed
    git add .                   - stage all changes
    git commit -m "message"     - commit staged changes
    git push                    - upload commits to GitHub
    git pull                    - download latest changes
    git log --oneline           - see commit history

Python/venv:
    python -m venv venv               - create virtual environment (Windows)
    python3.12 -m venv venv           - create virtual environment (Linux)
    venv\Scripts\activate             - activate (Windows)
    source venv/bin/activate          - activate (Linux)
    pip install -r requirements.txt   - install all dependencies
    deactivate                        - exit the virtual environment

Ollama:
    ollama pull llama3.1:8b     - download the model (one time)
    ollama list                 - see installed models
    ollama serve                - start the Ollama server manually if needed

Claude Code CLI:
    claude login                - sign in
    claude                       - start an interactive session in this folder
    claude --version            - check installed version

Testing:
    pytest tests/ -v             - run all tests with verbose output
    pytest tests/test_stt.py -v  - run one specific test file

--------------------------------------------------------------------------------
9A. COMMON pip install ERRORS AND FIXES
--------------------------------------------------------------------------------
"ERROR: No matching distribution found for argos-translate"
    -> Wrong package name. The correct name has NO hyphen: argostranslate
       Fix: pip install argostranslate>=1.9.4

"ERROR: No matching distribution found for TTS" (or a RuntimeError about
Python version on import)
    -> You installed/imported the old, unmaintained package. Use the
       maintained fork instead: pip install coqui-tts
       (the import in code stays "from TTS.api import TTS" - that part is
       correct and doesn't change)

"ModuleNotFoundError: No module named 'PyQt5.sip'" or similar PyQt5 errors
    -> PyQt5 doesn't have reliable Python 3.12 wheels. Use PySide6 instead:
       pip install PySide6
       (code must import from PySide6, not PyQt5 - see Phase 9 note above)

"Failed building wheel for webrtcvad"
    -> webrtcvad doesn't build on Python 3.12. Use silero-vad instead:
       pip install silero-vad

"Failed building wheel for sentencepiece" (can appear while installing
argostranslate, especially on Windows)
    -> Try installing sentencepiece by itself first, which pulls a prebuilt
       wheel: pip install sentencepiece
       then retry: pip install argostranslate>=1.9.4
       If it still fails on Windows, the most reliable fix is to do this
       step inside WSL2 (Windows Subsystem for Linux) or on a Linux machine.

If you hit an install error not listed here, copy the FULL error text into
your Claude Code session and ask it to identify the correct current PyPI
package name/version for Python 3.12 before trying random fixes.

--------------------------------------------------------------------------------
10. FINAL CHECKLIST BEFORE GOING LIVE
--------------------------------------------------------------------------------
[ ] All 10 phases (Section 7 of the design doc) built and individually tested
[ ] Confirmed "python --version" reports 3.12.x on every machine (dev and
    deployment) and that requirements.txt uses the exact package names in
    Section 3.4 of this README / Section 4 of the design doc (not the older
    TTS/PyQt5/webrtcvad/langchain names)
[ ] Restricted-topic filter tested adversarially - 100% catch rate (FR12)
[ ] RAG hallucination test - 0% fabricated answers on out-of-scope questions
[ ] Processing wait-prompt (FR14) fires within 300ms on every interaction
[ ] Full offline test - disconnect internet, confirm everything still works
[ ] 3-5 outside testers have used it live in English and Urdu
[ ] NAB compliance has approved: restricted-topics list, fallback wording,
    and the exact "please wait" prompt wording
[ ] Auto-start configured (systemd/Task Scheduler) on the deployment machine
[ ] Interaction logs (logs/interactions.db) verified to be complete and
    reviewable by NAB IT/compliance staff

================================================================================
END OF README
================================================================================
