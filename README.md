# JOY Compiler

一个基于MLIR的AI编译器框架。

## 概述

JOY 专注于支持 Qwen 0.6B 大语言模型。

### Dialect 重命名

- `joy`: 高层操作
- `joyl`: 低层内存操作  
- `joyh`: 硬件抽象层

## 构建

### 快速开始

```bash
cd joy
./scripts/init.sh
./scripts/build.sh
```

### 构建选项

```bash
./scripts/build.sh [--debug|--release] [--clean] [-j N]
```
