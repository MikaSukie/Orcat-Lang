Compiler Birthday: May 31st 2025
<div align="center">

<img src="https://github.com/user-attachments/assets/02dfcb6a-84e6-4954-b9a8-e911f462359f" width="200"/>

# 🐋 ORCat Compiler 🐱

[![Version](https://img.shields.io/badge/ORCatCompiler‑VER-B2.0-blue)](https://github.com/MikaLorielle/Orcat-Lang)
![GitHub last commit](https://img.shields.io/github/last-commit/MikaLorielle/Orcat-lang)
[![Issues](https://img.shields.io/github/issues/MikaLorielle/Orcat-Lang.svg)](https://github.com/MikaLorielle/Orcat-Lang/issues)
[![Stars](https://img.shields.io/github/stars/MikaLorielle/Orcat-Lang.svg?style=social)](https://github.com/MikaLorielle/Orcat-Lang/stargazers)

On Windows IMPORTANT you need the clang/llvm compiler. <br>
windows https://github.com/llvm/llvm-project/releases <br>
-and https://visualstudio.microsoft.com/visual-cpp-build-tools/ <br>
MacOS xcode-select --install <br>
Or use homebrew <br>
</div>

---

## ~> Project Status

**ORCat** is currently in **active development** **~~stopped/frozen~~**.  
While the compiler core is functional, work is in progress:

- # Standard library development  
- ## 🎨 Language feature polish  
- ## Runtime and performance enhancements

The syntax has been fully built and is being improved upon.  
Want to join the community? (Please do 🙏)

>> **[Join the Discord!](https://discord.gg/zmnuz4h88x)**

---
## Memory ownership (important) <br>
- many manual frees can be done via forget(varname) or the RAII-like autoregion {}. For libraries you may use the crumb system to limit reads and writes (optionally auto free only if both are specified.) <br>
- more coming soon.
---

## 👌 Goals

- This is a “small” but EXTREMELY ambitious compiler project.
- *(because I do not like how Rust did their safety and simplicity balance.),
- safety, and simplicity in a language, but unsafe if you need it to be.
- Full control by default, optional safety tools provided.
---

Stay tuned for more updates!🎉🎉
And thank you for checking out ORCat. 🐋🐱
