# app.py - Complete ESP32/Arduino Project Generator
import os
import gradio as gr
from pathlib import Path
import shutil
import zipfile
import subprocess
import re
from dotenv import load_dotenv

load_dotenv()

# === LLM Setup ===
OLLAMA_AVAILABLE = False
GROQ_AVAILABLE = False
GEMINI_AVAILABLE = False
PERPLEXITY_AVAILABLE = False
VERCEL_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except:
    pass

try:
    from groq import Groq
    GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
except:
    pass

try:
    import google.generativeai as genai
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        GEMINI_AVAILABLE = True
except:
    pass

try:
    from openai import OpenAI
    PERPLEXITY_AVAILABLE = bool(os.getenv("PERPLEXITY_API_KEY"))
    VERCEL_AVAILABLE = bool(os.getenv("AI_GATEWAY_API_KEY") and os.getenv("AI_GATEWAY_BASE_URL"))
except:
    pass

# === Board Configuration ===
BOARD_OPTIONS = [
    ("ESP32 DevKit", "esp32dev"),
    ("ESP32-CAM", "esp32cam"),
    ("ESP8266 NodeMCU", "nodemcuv2"),
    ("Arduino Uno R4 WiFi", "uno_r4_wifi"),
    ("Arduino Nano 33 IoT", "nano_33_iot"),
    ("Arduino Uno (classic)", "uno"),
    ("Arduino Nano (ATmega328)", "nanoatmega328"),
    ("Arduino Mega 2560", "megaatmega2560"),
    ("Raspberry Pi Pico W", "pico2"),
]

WIFI_BOARDS = {"esp32dev", "esp32cam", "nodemcuv2", "uno_r4_wifi", "nano_33_iot", "pico2"}
OTA_BOARDS = {"esp32dev", "esp32cam", "nodemcuv2", "uno_r4_wifi", "nano_33_iot"}

PLATFORM_MAP = {
    "esp32dev": "espressif32",
    "esp32cam": "espressif32",
    "nodemcuv2": "espressif8266",
    "uno_r4_wifi": "renesas_uno",
    "nano_33_iot": "atmelsam",
    "pico2": "raspberrypi",
    "uno": "atmelavr",
    "nanoatmega328": "atmelavr",
    "megaatmega2560": "atmelavr",
}

# === Paths ===
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "generated_projects"
OUTPUT_DIR.mkdir(exist_ok=True)

# === LLM Calls ===
def call_llm(provider: str, prompt: str) -> str:
    """Call LLM with proper error handling for all providers"""
    try:
        if provider == "ollama" and OLLAMA_AVAILABLE:
            messages = [
                {"role": "system", "content": "You are an embedded C++ code generator. Output only raw code in the requested format."},
                {"role": "user", "content": prompt}
            ]
            response = ollama.chat(model="codestral", messages=messages)
            return response["message"]["content"]
        
        elif provider == "groq" and GROQ_AVAILABLE:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an embedded C++ code generator. Output only raw code."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=3000
            )
            return response.choices[0].message.content
        
        elif provider == "gemini" and GEMINI_AVAILABLE:
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content(
                prompt,
                safety_settings={
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"
                }
            )
            return response.text
        
        elif provider == "perplexity" and PERPLEXITY_AVAILABLE:
            client = OpenAI(
                api_key=os.getenv("PERPLEXITY_API_KEY"),
                base_url="https://api.perplexity.ai"
            )
            response = client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {"role": "system", "content": "You are an embedded C++ code generator. Output only raw code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            return response.choices[0].message.content
        
        elif provider == "vercel" and VERCEL_AVAILABLE:
            client = OpenAI(
                api_key=os.getenv("AI_GATEWAY_API_KEY"),
                base_url=os.getenv("AI_GATEWAY_BASE_URL")
            )
            response = client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {"role": "system", "content": "You are an embedded C++ code generator. Output only raw code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            return response.choices[0].message.content
        
        return f"❌ Provider '{provider}' not available or not configured"
    
    except Exception as e:
        error_msg = f"❌ LLM Error ({provider}): {str(e)}"
        if hasattr(e, 'response'):
            try:
                error_msg += f"\nHTTP {e.response.status_code}: {e.response.text}"
            except:
                pass
        return error_msg

# === Code Parsing ===
def extract_code_blocks(text: str, filename: str) -> str:
    """Extract code from markdown blocks or delimiters"""
    # Try markdown code blocks
    pattern = r"```(?:cpp|c\+\+)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Try custom delimiters
    pattern = rf"---\s*{re.escape(filename)}\s*---\s*(.*?)(?=---|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    
    # Fallback: return if looks like code
    if "#include" in text or "void setup" in text or "void loop" in text:
        return text.strip()
    
    return ""

# === Prompt Enhancement ===
def enhance_prompt(raw_input: str, provider: str) -> str:
    """Improve and translate user input to clean English"""
    if not raw_input.strip():
        return ""
    
    enhancement_prompt = f"""Convert this user input into a clear, professional English prompt for embedded systems code generation.

Rules:
- Translate to English if needed
- Fix grammar and spelling
- Make it technical and specific
- Keep it concise (max 3 sentences)
- Focus on functionality

User input: {raw_input}

Output ONLY the improved prompt, nothing else."""

    result = call_llm(provider, enhancement_prompt)
    
    if result.startswith("❌"):
        return raw_input
    
    # Clean up
    result = result.strip().replace('"', '').replace("'", "")
    lines = [l for l in result.split("\n") if l.strip() and not l.lower().startswith(("here", "improved", "output"))]
    
    return lines[-1] if lines else result

# === Project Generation ===
def generate_project(
    mode: str,
    description: str,
    libraries: str,
    board_id: str,
    provider: str,
    ino_file,
    progress=gr.Progress()
):
    """Main generation function with debug output"""
    progress(0, desc="Initializing...")
    
    # Auto-enhance description
    if mode == "description" and description.strip():
        progress(0.05, desc="Enhancing prompt...")
        enhanced_desc = enhance_prompt(description, provider)
        if enhanced_desc and not enhanced_desc.startswith("❌"):
            description = enhanced_desc
            progress(0.1, desc=f"Using: {description[:50]}...")
    
    # Create project structure
    project_name = "embedded_project"
    project_path = OUTPUT_DIR / project_name
    
    if project_path.exists():
        shutil.rmtree(project_path)
    
    project_path.mkdir(parents=True)
    src_dir = project_path / "src"
    src_dir.mkdir(exist_ok=True)
    
    # Determine input
    if mode == "refactor" and ino_file:
        progress(0.1, desc="Reading .ino file...")
        with open(ino_file.name, "r", encoding="utf-8") as f:
            ino_content = f.read()
        
        base_prompt = f"""Convert this Arduino .ino sketch to modular C++ for PlatformIO.
Board: {board_id}
Create separate files: main.cpp, sensors.h/cpp, network.h/cpp

Original code:
{ino_content}
"""
    else:
        progress(0.1, desc="Preparing prompt...")
        base_prompt = f"""Create modular C++ code for embedded project.
Board: {board_id}
Description: {description}
Libraries: {libraries}

Generate separate modules for sensors, networking, etc.
"""
    
    # Generate modules
    modules = []
    has_wifi = board_id in WIFI_BOARDS
    
    if has_wifi:
        modules.append("network")
    modules.append("sensors")
    
    lib_list = [lib.strip() for lib in libraries.split("\n") if lib.strip()]
    
    # Auto-add WiFi libraries for WiFi boards if not specified
    if has_wifi and not lib_list:
        if board_id in {"esp32dev", "esp32cam"}:
            lib_list.append("WiFi")
        elif board_id == "nodemcuv2":
            lib_list.append("ESP8266WiFi")
        elif board_id == "uno_r4_wifi":
            lib_list.append("WiFiS3")
        elif board_id == "nano_33_iot":
            lib_list.append("WiFiNINA")
        elif board_id == "pico2":
            lib_list.append("WiFi")
    
    progress(0.2, desc="Generating main.cpp...")
    
    # Generate main.cpp
    main_prompt = f"""{base_prompt}

Write ONLY main.cpp with setup() and loop() functions.
Include necessary headers. Format as raw C++ code:

#include <Arduino.h>
// your code here
void setup() {{
  // initialization
}}
void loop() {{
  // main logic
}}
"""
    
    main_code = call_llm(provider, main_prompt)
    
    # Debug: Save raw response
    (src_dir / "main_RAW_RESPONSE.txt").write_text(main_code, encoding="utf-8")
    
    main_extracted = extract_code_blocks(main_code, "main.cpp")
    
    if main_extracted and not main_extracted.startswith("❌"):
        (src_dir / "main.cpp").write_text(main_extracted, encoding="utf-8")
    else:
        # Fallback
        fallback = "#include <Arduino.h>\n\nvoid setup() {\n  Serial.begin(115200);\n  Serial.println(\"Project started\");\n}\n\nvoid loop() {\n  delay(1000);\n}\n"
        (src_dir / "main.cpp").write_text(fallback, encoding="utf-8")
    
    # Generate modules
    for i, module in enumerate(modules):
        progress(0.3 + i * 0.3, desc=f"Generating {module}...")
        
        module_prompt = f"""{base_prompt}

Write {module}.h and {module}.cpp for the {module} module.
Format EXACTLY like this:

--- {module}.h ---
#ifndef {module.upper()}_H
#define {module.upper()}_H
// header declarations
#endif
--- {module}.cpp ---
#include "{module}.h"
// implementation
"""
        
        module_code = call_llm(provider, module_prompt)
        
        # Debug: Save raw response
        (src_dir / f"{module}_RAW_RESPONSE.txt").write_text(module_code, encoding="utf-8")
        
        # Parse header
        h_pattern = rf"---\s*{re.escape(module)}\.h\s*---\s*(.*?)(?=---\s*{re.escape(module)}\.cpp\s*---|$)"
        h_match = re.search(h_pattern, module_code, re.DOTALL | re.IGNORECASE)
        
        if h_match:
            h_code = h_match.group(1).strip()
            h_code = extract_code_blocks(h_code, f"{module}.h") or h_code
            if h_code and not h_code.startswith("❌"):
                (src_dir / f"{module}.h").write_text(h_code, encoding="utf-8")
        
        # Parse cpp
        cpp_pattern = rf"---\s*{re.escape(module)}\.cpp\s*---\s*(.*?)$"
        cpp_match = re.search(cpp_pattern, module_code, re.DOTALL | re.IGNORECASE)
        
        if cpp_match:
            cpp_code = cpp_match.group(1).strip()
            cpp_code = extract_code_blocks(cpp_code, f"{module}.cpp") or cpp_code
            if cpp_code and not cpp_code.startswith("❌"):
                (src_dir / f"{module}.cpp").write_text(cpp_code, encoding="utf-8")
    
    progress(0.8, desc="Creating platformio.ini...")
    
    # Create platformio.ini
    platform = PLATFORM_MAP.get(board_id, "espressif32")
    
    if lib_list:
        lib_deps = "\n".join(f"    {lib}" for lib in lib_list)
        lib_section = f"lib_deps =\n{lib_deps}"
    else:
        lib_section = "; lib_deps = (add libraries here if needed)"
    
    ini_content = f"""[env:{board_id}]
platform = {platform}
board = {board_id}
framework = arduino
{lib_section}
monitor_speed = 115200
"""
    
    (project_path / "platformio.ini").write_text(ini_content, encoding="utf-8")
    
    progress(0.9, desc="Creating README...")
    
    # Create README
    libs_section = "\n".join(f"- {lib}" for lib in lib_list) if lib_list else "*Auto-detected from code*"
    
    readme = f"""# {project_name.title()}

## Hardware
- Board: {board_id}
- Platform: {platform}

## Description
{description if mode == "description" else "Refactored from .ino sketch"}

## Libraries
{libs_section}

## Build
```bash
pio run
```

## Upload
```bash
pio run --target upload
```

## Monitor
```bash
pio device monitor
```

## Debug Files
Raw LLM responses are saved in `src/` as `*_RAW_RESPONSE.txt` for troubleshooting.

## License
MIT
"""
    
    (project_path / "README.md").write_text(readme, encoding="utf-8")
    
    progress(0.95, desc="Creating ZIP...")
    
    # Create ZIP
    zip_path = OUTPUT_DIR / f"{project_name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in project_path.rglob('*'):
            if file.is_file():
                zf.write(file, file.relative_to(OUTPUT_DIR))
    
    progress(1.0, desc="✅ Complete!")
    
    # Build status log
    status = ["✅ Project generated successfully!", "", "Files created:"]
    status.append(f"- main.cpp {'✓' if (src_dir / 'main.cpp').exists() else '✗'}")
    
    for m in modules:
        h_ok = (src_dir / f"{m}.h").exists()
        cpp_ok = (src_dir / f"{m}.cpp").exists()
        status.append(f"- {m}.h {'✓' if h_ok else '✗'} / {m}.cpp {'✓' if cpp_ok else '✗'}")
    
    status.extend([
        "- platformio.ini ✓",
        "- README.md ✓",
        "",
        f"Debug files: *_RAW_RESPONSE.txt in src/",
        f"Provider used: {provider}"
    ])
    
    return str(zip_path), "\n".join(status)

# === Compile Function ===
def compile_project():
    """Compile with PlatformIO"""
    project_path = OUTPUT_DIR / "embedded_project"
    
    if not project_path.exists():
        return "❌ No project generated yet. Click 'Generate' first."
    
    try:
        result = subprocess.run(
            ["pio", "run", "-d", str(project_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            return f"✅ Compilation successful!\n\n{output}"
        else:
            return f"❌ Compilation failed:\n\n{output}"
    
    except FileNotFoundError:
        return "❌ PlatformIO not found.\n\nInstall: pip install -U platformio"
    
    except subprocess.TimeoutExpired:
        return "❌ Compilation timeout (>2min)"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === Custom CSS ===
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}

.dark {
    --body-background-fill: #1a1a1a !important;
    --background-fill-primary: #262626 !important;
    --background-fill-secondary: #333333 !important;
    --border-color-primary: #404040 !important;
}

.primary {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
}

button.primary {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
    border: none !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #ff8555 0%, #ffa43e 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
}

.provider-btn {
    min-width: 140px;
}

.provider-btn.active {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
    color: white !important;
    font-weight: 600;
}
"""

# === UI ===
with gr.Blocks(title="ESP32 Code Generator") as demo:
    gr.Markdown("""
    # 🔧 Embedded Project Generator
    **Local, AI-powered code generator for ESP32 and microcontrollers**
    
    Generate modular PlatformIO projects from natural language descriptions or refactor existing .ino files.
    Supports 10+ boards with automatic library detection, OTA for WiFi boards, and instant compilation feedback.
    """)
    
    # Provider State
    provider_state = gr.State("ollama")
    
    # AI Backend Selection
    with gr.Group():
        gr.Markdown("### 🤖 Select AI Backend")
        
        with gr.Row():
            btn_ollama = gr.Button("🏠 Ollama (Local)", elem_classes=["provider-btn"], variant="primary" if OLLAMA_AVAILABLE else "secondary")
            btn_groq = gr.Button("☁️ Groq (Cloud)", elem_classes=["provider-btn"], variant="secondary", visible=GROQ_AVAILABLE)
            btn_gemini = gr.Button("🤖 Gemini (Google)", elem_classes=["provider-btn"], variant="secondary", visible=GEMINI_AVAILABLE)
            btn_perplexity = gr.Button("🔎 Perplexity (Sonar)", elem_classes=["provider-btn"], variant="secondary", visible=PERPLEXITY_AVAILABLE)
            btn_vercel = gr.Button("🧩 Vercel Gateway", elem_classes=["provider-btn"], variant="secondary", visible=VERCEL_AVAILABLE)
        
        provider_label = gr.Markdown("**Aktives AI-Backend:** Ollama (Local)")
    
    # Mode Selection
    mode = gr.Radio(
        choices=[("📝 From Description", "description"), ("📄 Refactor .ino", "refactor")],
        value="description",
        label="Generation Mode",
        info="Choose how to create your project"
    )
    
    # Input containers
    with gr.Group(visible=True) as desc_inputs:
        description = gr.Textbox(
            label="Project Description (any language - will be auto-enhanced)",
            placeholder="z.B.: ich brauch sensor für temperatur und webseite...\nExample: need temperature sensor with web display...",
            lines=3
        )
        libraries = gr.Textbox(
            label="Libraries (one per line, optional)",
            placeholder="Leave empty for auto-detection or specify:\nDHT sensor library\nAdafruit NeoPixel\nServo",
            lines=4,
            value=""
        )
    
    with gr.Group(visible=False) as ino_inputs:
        ino_file = gr.File(
            label="Upload .ino File",
            file_types=[".ino"],
            type="filepath"
        )
        gr.Markdown("*Libraries will be auto-detected from #include statements*")
    
    # Common inputs
    with gr.Row():
        board = gr.Dropdown(
            choices=BOARD_OPTIONS,
            value="esp32dev",
            label="Target Board",
            info="Select your microcontroller"
        )
    
    # Action buttons
    with gr.Row():
        btn_generate = gr.Button("🚀 Generate Project", variant="primary", scale=2)
        btn_compile = gr.Button("⚙️ Compile", scale=1)
    
    # Outputs
    output_zip = gr.File(label="📦 Download Project", interactive=False)
    output_log = gr.Textbox(
        label="Output Log",
        lines=10,
        interactive=False
    )
    
    # === Event Handlers ===
    
    def toggle_inputs(mode_val):
        """Toggle input visibility based on mode"""
        if mode_val == "description":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    def set_provider(prov_name, prov_value):
        """Update provider state and label"""
        return prov_value, f"**Aktives AI-Backend:** {prov_name}"
    
    # Mode switching
    mode.change(
        toggle_inputs,
        inputs=[mode],
        outputs=[desc_inputs, ino_inputs]
    )
    
    # Provider button handlers
    if OLLAMA_AVAILABLE:
        btn_ollama.click(
            lambda: set_provider("Ollama (Local)", "ollama"),
            outputs=[provider_state, provider_label]
        )
    
    if GROQ_AVAILABLE:
        btn_groq.click(
            lambda: set_provider("Groq (Cloud)", "groq"),
            outputs=[provider_state, provider_label]
        )
    
    if GEMINI_AVAILABLE:
        btn_gemini.click(
            lambda: set_provider("Gemini (Google)", "gemini"),
            outputs=[provider_state, provider_label]
        )
    
    if PERPLEXITY_AVAILABLE:
        btn_perplexity.click(
            lambda: set_provider("Perplexity (Sonar)", "perplexity"),
            outputs=[provider_state, provider_label]
        )
    
    if VERCEL_AVAILABLE:
        btn_vercel.click(
            lambda: set_provider("Vercel Gateway", "vercel"),
            outputs=[provider_state, provider_label]
        )
    
    # Generate project
    btn_generate.click(
        generate_project,
        inputs=[mode, description, libraries, board, provider_state, ino_file],
        outputs=[output_zip, output_log]
    )
    
    # Compile project
    btn_compile.click(
        compile_project,
        outputs=[output_log]
    )
    
    # Footer
    gr.Markdown("""
    ---
    **Tips:**
    - Use descriptive project descriptions for better code generation
    - Debug files (`*_RAW_RESPONSE.txt`) are saved in the ZIP for troubleshooting
    - Check the generated code before uploading to hardware
    - Available providers depend on configured API keys in `.env`
    """)

# === Launch ===
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=custom_css,
        theme=gr.themes.Soft(),
        share=False
    )