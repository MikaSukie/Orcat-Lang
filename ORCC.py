#!/usr/bin/env python3
# ORCC.py — ORCat Compiler

import sys
import re
from typing import List, Optional, Tuple, Union, Dict
from dataclasses import dataclass
TYPE_TOKENS = {'IDENT', 'INT', 'FLOAT', 'STRING', 'BOOL', 'CHAR', 'VOID'}
# ────────────────
# Token Definitions
# ────────────────
string_constants: List[str] = []

@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int

# Token types
KEYWORDS = {
    'fn', 'if', 'else', 'while', 'return',
    'import', 'pub', 'priv', 'prot', 'extern',
    'int', 'float', 'bool', 'char', 'string', 'void',
    'true', 'false'
}

SINGLE_CHARS = {
    '(': 'LPAREN', ')': 'RPAREN', '{': 'LBRACE', '}': 'RBRACE',
    ',': 'COMMA', ';': 'SEMI', ':': 'COLON', '=': 'EQUAL',
    '+': 'PLUS', '-': 'MINUS', '*': 'STAR', '/': 'SLASH',
    '<': 'LT', '>': 'GT'
}

MULTI_CHARS = {
    '==': 'EQEQ', '!=': 'NEQ', '<=': 'LE', '>=': 'GE', '->': 'ARROW'
}

def lex(source: str) -> List[Token]:
    tokens: List[Token] = []
    i, line, col = 0, 1, 1
    while i < len(source):
        c = source[i]

        if c in ' \t':
            i += 1
            col += 1
            continue
        if c == '\n':
            i += 1
            line += 1
            col = 1
            continue
        if c == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Multi-char operators
        for mc, kind in MULTI_CHARS.items():
            if source[i:i+len(mc)] == mc:
                tokens.append(Token(kind, mc, line, col))
                i += len(mc)
                col += len(mc)
                break
        else:
            # Identifiers, keywords
            if c.isalpha() or c == '_':
                start = i
                while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                    i += 1
                val = source[start:i]
                kind = val if val in KEYWORDS else 'IDENT'
                tokens.append(Token(kind.upper(), val, line, col))
                col += len(val)
                continue

            # Numbers (int, float)
            if c.isdigit():
                start = i
                while i < len(source) and source[i].isdigit():
                    i += 1
                if i < len(source) and source[i] == '.':
                    i += 1
                    while i < len(source) and source[i].isdigit():
                        i += 1
                    val = source[start:i]
                    tokens.append(Token('FLOAT', val, line, col))
                else:
                    val = source[start:i]
                    tokens.append(Token('INT', val, line, col))
                col += len(val)
                continue

            # Strings
            if c == '"':
                i += 1
                start = i
                while i < len(source) and source[i] != '"':
                    i += 1
                val = source[start:i]
                i += 1
                tokens.append(Token('STRING', val, line, col))
                col += len(val) + 2
                continue

            # Single-char tokens
            if c in SINGLE_CHARS:
                tokens.append(Token(SINGLE_CHARS[c], c, line, col))
                i += 1
                col += 1
                continue

            raise RuntimeError(f"Unrecognized character '{c}' at {line}:{col}")

    tokens.append(Token('EOF', '', line, col))
    return tokens
# ────────────────
# AST Definitions
# ────────────────

@dataclass
class Expr: pass
@dataclass
class Stmt: pass

# Expressions
@dataclass
class IntLit(Expr): value: int
@dataclass
class FloatLit(Expr): value: float
@dataclass
class BoolLit(Expr): value: bool
@dataclass
class StrLit(Expr): value: str
@dataclass
class Var(Expr): name: str
@dataclass
class BinOp(Expr): op: str; left: Expr; right: Expr
@dataclass
class Call(Expr): name: str; args: List[Expr]

# Statements
@dataclass
class VarDecl(Stmt):
    access: str
    typ: str
    name: str
    expr: Optional[Expr]

@dataclass
class Assign(Stmt): name: str; expr: Expr
@dataclass
class IfStmt(Stmt):
    cond: Expr
    then_body: List[Stmt]
    else_body: Optional[Union['IfStmt', List[Stmt]]]
class WhileStmt(Stmt): cond: Expr; body: List[Stmt]
@dataclass
class ReturnStmt(Stmt): expr: Optional[Expr]
@dataclass
class ExprStmt(Stmt): expr: Expr

@dataclass
class Func:
    access: str
    name: str
    params: List[Tuple[str, str]]  # List of (type, name)
    ret_type: str
    body: Optional[List[Stmt]] = None
    is_extern: bool = False

@dataclass
class Program:
    funcs: List[Func]
    imports: List[str]

# ────────────────
# Parser
# ────────────────

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def bump(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, kind: str) -> Token:
        if self.peek().kind == kind:
            return self.bump()
        raise SyntaxError(f"Expected {kind}, got {self.peek().kind} at {self.peek().line}:{self.peek().col}")

    def match(self, kind: str) -> bool:
        if self.peek().kind == kind:
            self.bump()
            return True
        return False

    def parse(self) -> Program:
        funcs = []
        imports = []
        while self.peek().kind != 'EOF':
            if self.match('IMPORT'):
                path = self.expect('STRING').value
                self.expect('SEMI')
                imports.append(path)
            else:
                funcs.append(self.parse_func())
        return Program(funcs, imports)

    def parse_func(self) -> Func:
        access = 'priv'
        is_extern = False
        if self.peek().kind == 'EXTERN':
            is_extern = True
            self.bump()
        elif self.peek().kind in {'PUB', 'PRIV', 'PROT'}:
            access = self.bump().kind.lower()

        self.expect('FN')
        name = self.expect('IDENT').value
        self.expect('LPAREN')
        params = []

        if self.peek().kind != 'RPAREN':
            while True:
                if self.peek().kind in {'IDENT', 'INT', 'FLOAT', 'STRING', 'BOOL', 'CHAR'}:
                    typ = self.bump().value
                else:
                    raise SyntaxError(f"Expected type in function parameter, got {self.peek().kind}")
                self.expect('COLON')
                pname = self.expect('IDENT').value
                params.append((typ, pname))
                if not self.match('COMMA'):
                    break

        self.expect('RPAREN')
        self.expect('LT')
        if self.peek().kind in {'IDENT', 'INT', 'FLOAT', 'STRING', 'BOOL', 'CHAR', 'VOID'}:
            ret_type = self.bump().value
        else:
            raise SyntaxError(f"Expected type after <>, got {self.peek().kind}")
        self.expect('GT')
        if is_extern:
            self.expect('SEMI')
            return Func(access, name, params, ret_type, None, True)
        else:
            self.expect('LBRACE')
            body = self.parse_block()
            self.expect('RBRACE')
            return Func(access, name, params, ret_type, body, False)

    def parse_block(self) -> List[Stmt]:
        stmts = []
        while self.peek().kind != 'RBRACE':
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self) -> Stmt:
        t = self.peek()

        if t.kind in {'PUB', 'PRIV', 'PROT'} or t.kind in {'INT', 'FLOAT', 'STRING', 'BOOL', 'CHAR'}:
            return self.parse_var_decl()
        if t.kind == 'IDENT' and self.tokens[self.pos + 1].kind == 'EQUAL':
            return self.parse_assign()
        if t.kind == 'IF':
            return self.parse_if()
        if t.kind == 'WHILE':
            return self.parse_while()
        if t.kind == 'RETURN':
            return self.parse_return()
        return self.parse_expr_stmt()

    def parse_var_decl(self) -> VarDecl:
        access = 'priv'
        if self.peek().kind in {'PUB', 'PRIV', 'PROT'}:
            access = self.bump().kind.lower()
        if self.peek().kind in {'IDENT', 'INT', 'FLOAT', 'STRING', 'BOOL', 'CHAR'}:
            typ = self.bump().value
        else:
            raise SyntaxError(f"Expected type in variable declaration, got {self.peek().kind}")
        name = self.expect('IDENT').value
        expr = None
        if self.match('EQUAL'):
            expr = self.parse_expr()
        self.expect('SEMI')
        return VarDecl(access, typ, name, expr)

    def parse_assign(self) -> Assign:
        name = self.expect('IDENT').value
        self.expect('EQUAL')
        expr = self.parse_expr()
        self.expect('SEMI')
        return Assign(name, expr)

    def parse_if(self) -> IfStmt:
        self.expect('IF')
        self.expect('LPAREN')
        cond = self.parse_expr()
        self.expect('RPAREN')
        self.expect('LBRACE')
        then_body = self.parse_block()
        self.expect('RBRACE')
        else_body = None
        if self.match('ELSE'):
            if self.peek().kind == 'IF':
                else_body = self.parse_if()
            else:
                self.expect('LBRACE')
                else_body = self.parse_block()
                self.expect('RBRACE')
        return IfStmt(cond, then_body, else_body)

    def parse_while(self) -> WhileStmt:
        self.expect('WHILE')
        self.expect('LPAREN')
        cond = self.parse_expr()
        self.expect('RPAREN')
        self.expect('LBRACE')
        body = self.parse_block()
        self.expect('RBRACE')
        return WhileStmt(cond, body)

    def parse_return(self) -> ReturnStmt:
        self.expect('RETURN')
        expr = None
        if self.peek().kind != 'SEMI':
            expr = self.parse_expr()
        self.expect('SEMI')
        return ReturnStmt(expr)

    def parse_expr_stmt(self) -> ExprStmt:
        expr = self.parse_expr()
        self.expect('SEMI')
        return ExprStmt(expr)

    def parse_expr(self) -> Expr:
        left = self.parse_primary()
        while self.peek().kind in {'PLUS', 'MINUS', 'STAR', 'SLASH', 'EQEQ', 'NEQ', 'LT', 'LE', 'GT', 'GE'}:
            op = self.bump().value
            right = self.parse_primary()
            left = BinOp(op, left, right)
        return left

    def parse_primary(self) -> Expr:
        t = self.bump()
        if t.kind == 'INT':
            return IntLit(int(t.value))
        if t.kind == 'FLOAT':
            return FloatLit(float(t.value))
        if t.kind == 'STRING':
            return StrLit(t.value)
        if t.kind == 'TRUE':
            return BoolLit(True)
        if t.kind == 'FALSE':
            return BoolLit(False)
        if t.kind == 'IDENT':
            if self.peek().kind == 'LPAREN':
                self.bump()
                args = []
                if self.peek().kind != 'RPAREN':
                    while True:
                        args.append(self.parse_expr())
                        if not self.match('COMMA'):
                            break
                self.expect('RPAREN')
                return Call(t.value, args)
            return Var(t.value)
        raise SyntaxError(f"Unexpected token: {t.kind} at {t.line}:{t.col}")
# ────────────────
# Code Generation (LLVM IR)
# ────────────────

tmp_id = 0
def new_tmp() -> str:
    global tmp_id
    tmp_id += 1
    return f"%t{tmp_id}"

label_id = 0
def new_label(base='L') -> str:
    global label_id
    label_id += 1
    return f"{base}{label_id}"

type_map = {
    'int': 'i32',
    'float': 'double',
    'bool': 'i1',
    'char': 'i8',
    'string': 'i8*',
    'void': 'void'
}

symbol_table: Dict[str, Tuple[str, str]] = {}
func_table: Dict[str, str] = {}

def gen_expr(expr: Expr, out: List[str]) -> str:
    global string_constants
    if isinstance(expr, IntLit):
        tmp = new_tmp()
        out.append(f"  {tmp} = add i32 0, {expr.value}")
        return tmp
    if isinstance(expr, FloatLit):
        tmp = new_tmp()
        out.append(f"  {tmp} = fadd double 0.0, {expr.value}")
        return tmp
    if isinstance(expr, BoolLit):
        tmp = new_tmp()
        val = 1 if expr.value else 0
        out.append(f"  {tmp} = add i1 0, {val}")
        return tmp
    if isinstance(expr, StrLit):
        tmp = new_tmp()
        label = f"@.str{len(string_constants)}"
        string_constants.append(
            f'{label} = private unnamed_addr constant [{len(expr.value)+1} x i8] c"{expr.value}\\00"'
        )
        out.append(f"  {tmp} = getelementptr inbounds [{len(expr.value)+1} x i8], [{len(expr.value)+1} x i8]* {label}, i32 0, i32 0")
        return tmp
    if isinstance(expr, Var):
        if expr.name not in symbol_table:
            raise RuntimeError(f"Undefined variable: {expr.name}")
        typ, name = symbol_table[expr.name]
        tmp = new_tmp()
        out.append(f"  {tmp} = load {typ}, {typ}* %{name}_addr")
        return tmp
    if isinstance(expr, BinOp):
        lhs = gen_expr(expr.left, out)
        rhs = gen_expr(expr.right, out)
        ty = infer_type(expr.left)
        llvm_ty = type_map[ty]
        tmp = new_tmp()
        op = {
            '+': 'add', '-': 'sub', '*': 'mul', '/': 'sdiv',
            '==': 'icmp eq', '!=': 'icmp ne',
            '<': 'icmp slt', '<=': 'icmp sle', '>': 'icmp sgt', '>=': 'icmp sge'
        }.get(expr.op)
        if llvm_ty == 'double':
            op = {
                '+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv',
                '==': 'fcmp oeq', '!=': 'fcmp one',
                '<': 'fcmp olt', '<=': 'fcmp ole', '>': 'fcmp ogt', '>=': 'fcmp oge'
            }.get(expr.op)
        out.append(f"  {tmp} = {op} {llvm_ty} {lhs}, {rhs}")
        return tmp
    if isinstance(expr, Call):
        args_ir = []
        for arg in expr.args:
            a = gen_expr(arg, out)
            ty = type_map[infer_type(arg)]
            args_ir.append(f"{ty} {a}")
        ret_ty = func_table.get(expr.name, 'i32')
        if ret_ty == 'void':
            out.append(f"  call void @{expr.name}({', '.join(args_ir)})")
            return ''  # no value to return
        else:
            tmp = new_tmp()
            out.append(f"  {tmp} = call {ret_ty} @{expr.name}({', '.join(args_ir)})")
            return tmp

    raise RuntimeError(f"Unhandled expr: {expr}")

def infer_type(expr: Expr) -> str:
    if isinstance(expr, IntLit): return 'int'
    if isinstance(expr, FloatLit): return 'float'
    if isinstance(expr, BoolLit): return 'bool'
    if isinstance(expr, StrLit): return 'string'
    if isinstance(expr, Var):
        if expr.name in symbol_table:
            return next(k for k in type_map if type_map[k] == symbol_table[expr.name][0])
    if isinstance(expr, BinOp):
        return infer_type(expr.left)
    if isinstance(expr, Call):
        return next(k for k in type_map if type_map[k] == func_table.get(expr.name, 'i32'))
    return 'int'

def gen_stmt(stmt: Stmt, out: List[str]):
    if isinstance(stmt, VarDecl):
        llvm_ty = type_map[stmt.typ]
        out.append(f"  %{stmt.name}_addr = alloca {llvm_ty}")
        symbol_table[stmt.name] = (llvm_ty, stmt.name)
        if stmt.expr:
            val = gen_expr(stmt.expr, out)
            out.append(f"  store {llvm_ty} {val}, {llvm_ty}* %{stmt.name}_addr")
    elif isinstance(stmt, Assign):
        val = gen_expr(stmt.expr, out)
        llvm_ty, _ = symbol_table[stmt.name]
        out.append(f"  store {llvm_ty} {val}, {llvm_ty}* %{stmt.name}_addr")
    elif isinstance(stmt, IfStmt):
        cond = gen_expr(stmt.cond, out)
        then_lbl = new_label('then')
        else_lbl = new_label('else') if stmt.else_body else None
        end_lbl = new_label('endif')
        out.append(f"  br i1 {cond}, label %{then_lbl}, label %{else_lbl or end_lbl}")
        out.append(f"{then_lbl}:")
        for s in stmt.then_body:
            gen_stmt(s, out)
        out.append(f"  br label %{end_lbl}")
        if stmt.else_body:
            out.append(f"{else_lbl}:")
            if isinstance(stmt.else_body, list):
                for s in stmt.else_body:
                    gen_stmt(s, out)
            elif isinstance(stmt.else_body, IfStmt):
                gen_stmt(stmt.else_body, out)
            out.append(f"  br label %{end_lbl}")
        out.append(f"{end_lbl}:")
    elif isinstance(stmt, WhileStmt):
        head_lbl = new_label('while')
        body_lbl = new_label('body')
        end_lbl = new_label('endwhile')
        out.append(f"  br label %{head_lbl}")
        out.append(f"{head_lbl}:")
        cond = gen_expr(stmt.cond, out)
        out.append(f"  br i1 {cond}, label %{body_lbl}, label %{end_lbl}")
        out.append(f"{body_lbl}:")
        for s in stmt.body:
            gen_stmt(s, out)
        out.append(f"  br label %{head_lbl}")
        out.append(f"{end_lbl}:")
    elif isinstance(stmt, ReturnStmt):
        val = gen_expr(stmt.expr, out) if stmt.expr else None
        if val:
            llvm_ty = type_map[infer_type(stmt.expr)]
            out.append(f"  ret {llvm_ty} {val}")
        else:
            out.append(f"  ret void")
    elif isinstance(stmt, ExprStmt):
        gen_expr(stmt.expr, out)

def gen_func(fn: Func) -> List[str]:
    if fn.is_extern:
        param_sig = ", ".join(f"{type_map[t]}" for t, _ in fn.params)
        ret_ty = type_map[fn.ret_type]
        func_table[fn.name] = ret_ty
        return [f"declare {ret_ty} @{fn.name}({param_sig})"]
    symbol_table.clear()
    param_sig = ", ".join(f"{type_map[t]} %{n}" for t, n in fn.params)
    ret_ty = type_map[fn.ret_type]
    func_table[fn.name] = ret_ty
    out = [f"define {ret_ty} @{fn.name}({param_sig}) {{", "entry:"]
    for typ, name in fn.params:
        llvm_ty = type_map[typ]
        out.append(f"  %{name}_addr = alloca {llvm_ty}")
        out.append(f"  store {llvm_ty} %{name}, {llvm_ty}* %{name}_addr")
        symbol_table[name] = (llvm_ty, name)
    for stmt in fn.body:
        gen_stmt(stmt, out)
    if fn.ret_type == 'int':
        out.append("  ret i32 0")
    elif fn.ret_type == 'float':
        out.append("  ret double 0.0")
    elif fn.ret_type == 'bool':
        out.append("  ret i1 0")
    elif fn.ret_type == 'void':
        out.append("  ret void")
    out.append("}")
    return out

def compile_program(prog: Program) -> str:
    string_constants.clear()
    lines = []

    for imp in prog.imports:
        with open(imp, 'r') as f:
            src = f.read()
        tokens = lex(src)
        parser = Parser(tokens)
        sub_prog = parser.parse()
        for f in sub_prog.funcs:
            lines += gen_func(f)

    for fn in prog.funcs:
        lines += gen_func(fn)

    # Final output with string constants + header
    return '\n'.join([
        "; ModuleID = 'orcat'",
        'source_filename = "main.orcat"',
        '',
        *string_constants,
        *lines
    ])

# ────────────────
# Main Entrypoint
# ────────────────
def main():
    if len(sys.argv) != 4 or sys.argv[2] != "-o":
        print("Usage: ORCC.py <input.sorcat|.orcat> -o <output.ll>")
        return
    inp = sys.argv[1]
    outp = sys.argv[3]
    with open(inp) as f:
        src = f.read()
    tokens = lex(src)
    parser = Parser(tokens)
    prog = parser.parse()
    llvm = compile_program(prog)
    with open(outp, 'w') as f:
        f.write(llvm)

if __name__ == "__main__":
    main()
