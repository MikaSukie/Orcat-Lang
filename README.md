<div align="center">

<img src="https://github.com/user-attachments/assets/02dfcb6a-84e6-4954-b9a8-e911f462359f" width="200"/>

# 🐋 ORCat Compiler 🐱

[![Version](https://img.shields.io/badge/ORCatCompiler‑VER-1.8‑Beta-blue)](https://github.com/MikaLorielle/Orcat-Lang)
![GitHub last commit](https://img.shields.io/github/last-commit/MikaLorielle/Orcat-lang)
[![Issues](https://img.shields.io/github/issues/MikaLorielle/Orcat-Lang.svg)](https://github.com/MikaLorielle/Orcat-Lang/issues)
[![Stars](https://img.shields.io/github/stars/MikaLorielle/Orcat-Lang.svg?style=social)](https://github.com/MikaLorielle/Orcat-Lang/stargazers)

On Windows IMPORTANT you need the CLANG compiler. <br>
windows https://github.com/llvm/llvm-project/releases <br>
- and https://visualstudio.microsoft.com/visual-cpp-build-tools/ <br>
MacOS xcode-select --install <br>
Or use homebrew <br>
</div>

---

## 🚧 Project Status
🐋
**ORCat** is currently in **active development** **~~Slowed~~**.  
While the compiler core is functional, work is ongoing on:

- 🧱 Standard library development  
- 🧬 Language feature polish  
- ⚙️ Runtime and performance enhancements

The syntax has been fully built and is being iterated on.  
Want to be part of this?

👉 **[Join the Discord!](https://discord.gg/zmnuz4h88x)**

---
## Memory ownership (important)

- Many stdlib functions return heap-allocated C strings (e.g. `read_file`, `input`, `tostring`, `sb_*`, `i64tostr`, `ftostr`). These are owned by the caller and **must** be freed via `free_str()` in Orcat.
- Some helpers (like `get_os()` / `get_os_max_bits()` in older versions) returned static pointers. After the 2025-10-31 patch, all string-returning functions used by Orcat return allocated strings to avoid undefined behavior — callers can `free_str()` safely.
- Also IMPORTANT. If you are going to append string literals into a print or any C externed functions, note that string literals are stored in static memory.
- Basically either use in this format print(empty() + whatever_else); or assign the string like string s = safestring("test"); which will auto allocate and copy into the heap (modifiable).
---

## 📌 Notes

- This is a “small” but VERY ambitious compiler project.
- *(because I do not like compiled languages that slap the dev for doing something risky),
- safety, and simplicity in a language.

---

Stay tuned for more updates! 🌟  
And thank you for checking out ORCat. 🐋🐱
