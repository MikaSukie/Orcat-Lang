'''
 * This file is licensed under the GPL-3 License (or AGPL-3 if applicable)
 * Copyright (C) 2025  MikaSukie (old user), MikaLorielle (alt user), EmikaMai (current user), JaydenFreeman (legal name)
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import re, os, tomllib, argparse
from typing import List, Optional, Tuple, Union, Dict
from dataclasses import dataclass
compiled=""
builtins_emitted = False
extension_registry = {
    "llvm.globals": [],
    "registrations": {}
}
TYPE_TOKENS = {
    'IDENT', 'INT', 'INT8', 'INT16', 'INT32', 'INT64',
    'FLOAT', 'STRING', 'BOOL', 'CHAR', 'VOID'
    }
@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int
KEYWORDS = {
    'fn', 'if', 'else', 'while', 'return',
    'import', 'pub', 'priv', 'prot', 'extern',
    'int', 'int8', 'int16', 'int32', 'int64',
    'float', 'bool', 'char', 'string', 'void',
    'true', 'false', 'struct', 'enum', 'match', 'nomd',
    'pin', 'crumble', 'null', 'continue', 'break'
    }
SINGLE_CHARS = {
    '(': 'LPAREN',   ')': 'RPAREN',   '{': 'LBRACE',   '}': 'RBRACE',
    ',': 'COMMA',    ';': 'SEMI',
    '=': 'EQUAL',    '+': 'PLUS',     '-': 'MINUS',    '*': 'STAR',
    '/': 'SLASH',    '<': 'LT',       '>': 'GT',       '[': 'LBRACKET',
    ']': 'RBRACKET', '?': 'QUESTION', '.': 'DOT', ':': 'COLON', '%': 'PERCENT', '!': 'BANG'
    }
MULTI_CHARS = {
    '==': 'EQEQ', '!=': 'NEQ', '<=': 'LE', '>=': 'GE', '->': 'ARROW', '&&': 'AND',  '||': 'OR',
    '+=': 'PLUSEQ', '-=': 'MINUSEQ', '*=': 'STAREQ', '/=': 'SLASHEQ', '%=': 'PERCENTEQ',
    '&=': 'ANDEQ', '|=': 'OREQ', '^=': 'XOREQ', '<<=': 'LSHIFTEQ', '>>=': 'RSHIFTEQ',
    }
class SymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str, Tuple[str, str]]] = [{}]
    def push(self):
        self.scopes.append({})
    def pop(self):
        self.scopes.pop()
    def declare(self, name: str, llvm_type: str, ir_name: str):
        self.scopes[-1][name] = (llvm_type, ir_name)
    def lookup(self, name: str) -> Optional[Tuple[str, str]]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    def current(self) -> Dict[str, Tuple[str, str]]:
        return self.scopes[-1]
class TypeEnv:
    def __init__(self):
        self.scopes: List[Dict[str, str]] = [{}]
    def push(self):
        self.scopes.append({})
    def pop(self):
        self.scopes.pop()
    def declare(self, name: str, typ: str):
        self.scopes[-1][name] = typ
    def lookup(self, name: str) -> Optional[str]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
def llvm_type_of(typ: str) -> str:
    return type_map.get(typ, f"%struct.{typ}")
def clean_struct_name(name: str) -> str:
    return name.rstrip("*").removeprefix("%struct.")
def llvm_int_bitsize(ty: str) -> Optional[int]:
    m = re.fullmatch(r'i(\d+)', ty)
    if m:
        return int(m.group(1))
    return None
def extract_array_base_type(llvm_ty: str) -> str:
    match = re.match(r'\[\d+\s*x\s+(.+)\]', llvm_ty)
    if not match:
        raise RuntimeError(f"Cannot extract element type from: {llvm_ty}")
    return match.group(1)
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
        if c == '/' and i + 1 < len(source) and source[i + 1] == '*':
            i += 2
            while i < len(source) - 1:
                if source[i] == '*' and source[i + 1] == '/':
                    i += 2
                    break
                if source[i] == '\n':
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1
            else:
                raise RuntimeError(f"Unclosed multiline comment starting at line {line}")
            continue
        matched = False
        for mc, kind in MULTI_CHARS.items():
            if source.startswith(mc, i):
                tokens.append(Token(kind, mc, line, col))
                i += len(mc)
                col += len(mc)
                matched = True
                break
        if matched:
            continue
        if c.isalpha() or c == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            val = source[start:i]
            kind = val if val in KEYWORDS else 'IDENT'
            tokens.append(Token(kind.upper(), val, line, col))
            col += len(val)
            continue
        if c.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            is_float = False
            if i < len(source) and source[i] == '.':
                i += 1
                while i < len(source) and source[i].isdigit():
                    i += 1
                is_float = True
            if i < len(source) and source[i] in {'e', 'E'}:
                i += 1
                if i < len(source) and source[i] in {'+', '-'}:
                    i += 1
                while i < len(source) and source[i].isdigit():
                    i += 1
                is_float = True
            if i < len(source) and source[i] in {'f', 'F'}:
                i += 1
                is_float = True
            val = source[start:i]
            if is_float:
                tokens.append(Token('FLOAT', val, line, col))
            else:
                tokens.append(Token('INT', val, line, col))
            col += len(val)
            continue
        if c == '"':
            i += 1
            start_col = col
            val = ""
            while i < len(source) and source[i] != '"':
                if source[i] == '\\' and i + 1 < len(source):
                    nxt = source[i+1]
                    if nxt == 'n':
                        val += '\n'
                    elif nxt == 't':
                        val += '\t'
                    elif nxt == '\\':
                        val += '\\'
                    elif nxt == '"':
                        val += '"'
                    else:
                        val += nxt
                    i += 2
                    col += 2
                else:
                    val += source[i]
                    i += 1
                    col += 1
            if i >= len(source) or source[i] != '"':
                raise RuntimeError(f"Unclosed string literal at {line}:{start_col}")
            i += 1
            col += 1
            tokens.append(Token('STRING', val, line, start_col))
            continue
        if c == "'":
            i += 1
            start_col = col
            if i < len(source) and source[i] == '\\' and i + 1 < len(source):
                nxt = source[i+1]
                if nxt == 'n':
                    val = '\n'
                elif nxt == 't':
                    val = '\t'
                elif nxt == '\\':
                    val = '\\'
                elif nxt == "'":
                    val = "'"
                else:
                    val = nxt
                i += 2
                col += 2
            else:
                if i < len(source):
                    val = source[i]
                    i += 1
                    col += 1
                else:
                    raise RuntimeError(f"Unclosed character literal at {line}:{start_col}")
            if i >= len(source) or source[i] != "'":
                raise RuntimeError(f"Unclosed character literal at {line}:{start_col}")
            i += 1
            col += 1
            tokens.append(Token('CHAR', val, line, start_col))
            continue
        if c in SINGLE_CHARS:
            tokens.append(Token(SINGLE_CHARS[c], c, line, col))
            i += 1
            col += 1
            matched = True
            continue
        raise RuntimeError(f"Unrecognized character '{c}' at {line}:{col}")
    tokens.append(Token('EOF', '', line, col))
    return tokens
@dataclass
class Expr: pass
@dataclass
class Stmt: pass
@dataclass
class IntLit(Expr): value: int
@dataclass
class FloatLit(Expr): value: float
@dataclass
class BoolLit(Expr): value: bool
@dataclass
class CharLit(Expr): value: str
@dataclass
class StrLit(Expr): value: str
@dataclass
class Var(Expr): name: str
@dataclass
class NullLit(Expr): pass
@dataclass
class GlobalVar:
    typ: str
    name: str
    expr: Optional[Expr]
    nomd: bool = False
    pinned: bool = False
@dataclass
class DerefStmt(Stmt):
    varname: str
@dataclass
class UnaryDeref(Expr):
    ptr: Expr
@dataclass
class BinOp(Expr): op: str; left: Expr; right: Expr
@dataclass
class Call(Expr): name: str; args: List[Expr]
@dataclass
class EnumVariant:
    name: str
    typ: Optional[str]
@dataclass
class EnumDef(Stmt):
    name: str
    type_param: Optional[str]
    variants: List[EnumVariant]
@dataclass
class CrumbleStmt(Stmt):
    name: str
    max_reads: Optional[int] = None
    max_writes: Optional[int] = None
@dataclass
class VarDecl(Stmt):
    access: str
    typ: str
    name: str
    expr: Optional[Expr]
    nomd: bool = False
@dataclass
class Assign(Stmt):
    name: Union[str, Expr]
    expr: Expr
@dataclass
class StructField:
    name: str
    typ: str
@dataclass
class IndexAssign(Stmt):
    array: str
    index: Expr
    value: Expr
@dataclass
class StructDef(Stmt):
    name: str
    fields: List[StructField]
@dataclass
class FieldAccess(Expr):
    base: Expr
    field: str
@dataclass
class MatchCase:
    variant: str
    binding: Optional[str]
    body: List[Stmt]
@dataclass
class Match(Stmt):
    expr: Expr
    cases: List[MatchCase]
@dataclass
class StructInit(Expr):
    name: str
    fields: List[Tuple[str, Expr]]
@dataclass
class IfStmt(Stmt):
    cond: Expr
    then_body: List[Stmt]
    else_body: Optional[Union['IfStmt', List[Stmt]]]
@dataclass
class WhileStmt(Stmt):
    cond: Expr
    body: List[Stmt]
@dataclass
class ReturnStmt(Stmt):
    expr: Optional[Expr]
@dataclass
class ExprStmt(Stmt): expr: Expr
@dataclass
class Index(Expr):
    array: Expr
    index: Expr
@dataclass
class TypeofExpr(Expr):
    kind: str
    expr: Expr
@dataclass
class ContinueStmt(Stmt):
    pass
@dataclass
class BreakStmt(Stmt):
    pass
@dataclass
class Ternary(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr
@dataclass
class Func:
    access: str
    name: str
    type_params: List[str]
    params: List[Tuple[str, str]]
    ret_type: str
    body: Optional[List[Stmt]] = None
    is_extern: bool = False
@dataclass
class Program:
    funcs: List[Func]
    imports: List[str]
    structs: List[StructDef]
    enums: List[EnumDef]
    globals: List[GlobalVar]
string_constants: List[str] = []
struct_field_map: Dict[str, List[Tuple[str, str]]] = {}
generated_mono: Dict[str, bool] = {}
all_funcs: List[Func] = []
enum_variant_map: Dict[str, List[Tuple[str, Optional[str]]]] = {}
loop_stack: List[Dict[str, str]] = []
class Parser:
    def __init__(self, tokens: List[Token]):
        self.declared_vars: Dict[str, VarDecl] = {}
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
        structs = []
        enums = []
        globals = []
        while self.peek().kind != 'EOF':
            if self.match('IMPORT'):
                while True:
                    if self.peek().kind == 'STRING':
                        raw = self.bump().value
                    elif self.peek().kind == 'IDENT':
                        raw = self.bump().value
                    else:
                        raise SyntaxError(f"Expected import path, got {self.peek().kind} at {self.peek().line}:{self.peek().col}")
                    imports.append(raw)
                    if self.peek().kind == 'SEMI':
                        self.bump()
                        break
                    elif self.peek().kind == 'COMMA':
                        self.bump()
                        continue
                    else:
                        raise SyntaxError(f"Expected ',' or ';' in import list, got {self.peek().kind} at {self.peek().line}:{self.peek().col}")
            elif self.peek().kind in {'NOMD', 'PIN'}:
                nomd = False
                pinned = False
                while self.peek().kind in {'NOMD', 'PIN'}:
                    if self.match('NOMD'):
                        nomd = True
                    elif self.match('PIN'):
                        pinned = True
                decl = self.parse_var_decl()
                globals.append(GlobalVar(decl.typ, decl.name, decl.expr, nomd=nomd, pinned=pinned))
            elif self.peek().kind == 'STRUCT':
                structs.append(self.parse_struct_def())
            elif self.peek().kind == 'ENUM':
                enums.append(self.parse_enum_def())
            else:
                funcs.append(self.parse_func())
        self.program = Program(funcs, imports, structs, enums, globals)
        return self.program
    def parse_global(self) -> GlobalVar:
        if self.peek().kind not in TYPE_TOKENS and self.peek().kind != 'IDENT':
            raise SyntaxError(f"Expected type after 'pin', got {self.peek().kind}")
        typ = self.bump().value
        name = self.expect('IDENT').value
        expr = None
        if self.match('EQUAL'):
            expr = self.parse_expr()
        self.expect('SEMI')
        return GlobalVar(typ, name, expr)
    def parse_enum_def(self) -> EnumDef:
        self.expect('ENUM')
        name = self.expect('IDENT').value
        type_param: Optional[str] = None
        if self.match('LT'):
            type_param = self.expect('IDENT').value
            self.expect('GT')
        self.expect('LBRACE')
        variants: List[EnumVariant] = []
        while self.peek().kind != 'RBRACE':
            variant_name = self.expect('IDENT').value
            variant_type: Optional[str] = None
            if self.match('COLON'):
                variant_type = self.expect('IDENT').value
            self.expect('SEMI')
            variants.append(EnumVariant(variant_name, variant_type))
        self.expect('RBRACE')
        return EnumDef(name, type_param, variants)
    def parse_struct_def(self) -> StructDef:
        self.expect('STRUCT')
        name = self.expect('IDENT').value
        self.expect('LBRACE')
        fields = []
        while self.peek().kind != 'RBRACE':
            if self.peek().kind in TYPE_TOKENS:
                typ = self.bump().value
                fname = self.expect('IDENT').value
            else:
                raise SyntaxError(
                    f"Expected type in struct field, got {self.peek().kind} at {self.peek().line}:{self.peek().col}")
            self.expect('SEMI')
            fields.append(StructField(fname, typ))
        self.expect('RBRACE')
        return StructDef(name, fields)
    def parse_func(self) -> Func:
        access = 'pub'
        is_extern = False
        modifiers = []
        while True:
            tk = self.peek().kind
            if tk == 'EXTERN':
                is_extern = True
                self.bump()
                continue
            if tk in {'PUB', 'PRIV', 'PROT'}:
                access = self.bump().kind.lower()
                continue
            if tk != 'FN' and tk.lower() in KEYWORDS:
                modifiers.append(self.bump().value)
                continue
            break
        self.expect('FN')
        type_params: List[str] = []
        if self.peek().kind == 'LT':
            self.bump()
            while True:
                type_params.append(self.expect('IDENT').value)
                if not self.match('COMMA'):
                    break
            self.expect('GT')
        name = self.expect('IDENT').value
        self.expect('LPAREN')
        params: List[Tuple[str, str]] = []
        if self.peek().kind != 'RPAREN':
            while True:
                if self.peek().kind in TYPE_TOKENS or self.peek().kind == 'IDENT':
                    typ = self.bump().value
                    if self.match('STAR'):
                        typ += '*'
                    pname = self.expect('IDENT').value
                    params.append((typ, pname))
                else:
                    raise SyntaxError(
                        f"Expected type, got {self.peek().kind} "
                        f"at {self.peek().line}:{self.peek().col}")
                if not self.match('COMMA'):
                    break
        self.expect('RPAREN')
        self.expect('LT')
        ret_type = self.bump().value
        if self.match('STAR'):
            ret_type += '*'
        self.expect('GT')
        if is_extern:
            self.expect('SEMI')
            return Func(access, name, type_params, params, ret_type, None, True)
        self.expect('LBRACE')
        body = self.parse_block()
        self.expect('RBRACE')
        return Func(access, name, type_params, params, ret_type, body, False)
    def parse_block(self) -> List[Stmt]:
        stmts = []
        while self.peek().kind != 'RBRACE':
            stmts.append(self.parse_stmt())
        return stmts
    def parse_stmt(self) -> Stmt:
        t = self.peek()
        if t.kind == 'MATCH':
            return self.parse_match()
        if t.kind == 'STAR':
            return self.parse_ptr_assign()
        if t.kind == 'IDENT':
            next_kind = self.tokens[self.pos + 1].kind
            if next_kind in {
                'EQUAL', 'PLUSEQ', 'MINUSEQ', 'STAREQ', 'SLASHEQ',
                'PERCENTEQ', 'ANDEQ', 'OREQ', 'XOREQ', 'LSHIFTEQ', 'RSHIFTEQ'
            }:
                return self.parse_compound_assign()
        if t.kind == 'IDENT' and self.tokens[self.pos + 1].kind == 'LBRACKET':
            return self.parse_index_assign()
        if t.kind == 'IDENT' and t.value == 'deref' and self.tokens[self.pos + 1].kind == 'LPAREN':
            return self.parse_deref()
        if t.kind == 'IDENT' and self.tokens[self.pos + 1].kind == 'LPAREN':
            return self.parse_expr_stmt()
        if t.kind in {'PUB', 'PRIV', 'PROT', 'NOMD'} or t.kind in TYPE_TOKENS or t.kind == 'IDENT':
            return self.parse_var_decl()
        if t.kind == 'IF':
            return self.parse_if()
        if t.kind == 'WHILE':
            return self.parse_while()
        if self.match('CRUMBLE'):
            return self.parse_crumble()
        if t.kind == 'CONTINUE':
            self.bump()
            self.expect('SEMI')
            return ContinueStmt()
        if t.kind == 'BREAK':
            self.bump()
            self.expect('SEMI')
            return BreakStmt()
        if t.kind == 'RETURN':
            return self.parse_return()
        return self.parse_expr_stmt()
    def parse_compound_assign(self) -> Assign:
        name = self.expect('IDENT').value
        op_token = self.bump()
        expr = self.parse_expr()
        self.expect('SEMI')
        if op_token.kind == 'EQUAL':
            return Assign(name, expr)
        compound_map = {
            'PLUSEQ': '+',
            'MINUSEQ': '-',
            'STAREQ': '*',
            'SLASHEQ': '/',
            'PERCENTEQ': '%',
            'ANDEQ': '&',
            'OREQ': '|',
            'XOREQ': '^',
            'LSHIFTEQ': '<<',
            'RSHIFTEQ': '>>'
        }
        if op_token.kind not in compound_map:
            raise SyntaxError(f"Unknown compound assignment: {op_token.kind}")
        op = compound_map[op_token.kind]
        lhs_var = Var(name)
        binop = BinOp(op, lhs_var, expr)
        return Assign(name, binop)
    def parse_crumble(self) -> CrumbleStmt:
        self.expect('LPAREN')
        var_name = self.expect('IDENT').value
        self.expect('RPAREN')
        max_r, max_w = None, None
        while self.match('BANG'):
            kw = self.expect('IDENT').value
            self.expect('EQUAL')
            val = int(self.expect('INT').value)
            if kw == 'r':
                max_r = val
            elif kw == 'w':
                max_w = val
            else:
                raise SyntaxError(f"Unknown crumb kind '!{kw}'")
        self.expect('SEMI')
        return CrumbleStmt(var_name, max_r, max_w)
    def parse_ptr_assign(self) -> Stmt:
        self.expect('STAR')
        ptr_expr = self.parse_primary()
        self.expect('EQUAL')
        val_expr = self.parse_expr()
        self.expect('SEMI')
        return Assign(UnaryDeref(ptr_expr), val_expr)
    def parse_deref(self) -> DerefStmt:
        self.expect('IDENT')
        self.expect('LPAREN')
        varname = self.expect('IDENT').value
        self.expect('RPAREN')
        self.expect('SEMI')
        return DerefStmt(varname)
    def parse_match(self) -> Match:
        self.expect('MATCH')
        self.expect('LPAREN')
        expr_to_match = self.parse_expr()
        self.expect('RPAREN')
        self.expect('LBRACE')
        cases: List[MatchCase] = []
        while self.peek().kind != 'RBRACE':
            variant_name = self.expect('IDENT').value
            binding_name: Optional[str] = None
            if self.match('LPAREN'):
                binding_name = self.expect('IDENT').value
                self.expect('RPAREN')
            self.expect('COLON')
            self.expect('LBRACE')
            body_stmts: List[Stmt] = []
            while self.peek().kind != 'RBRACE':
                body_stmts.append(self.parse_stmt())
            self.expect('RBRACE')
            cases.append(MatchCase(variant_name, binding_name, body_stmts))
        self.expect('RBRACE')
        return Match(expr_to_match, cases)
    def parse_var_decl(self) -> VarDecl:
        access = 'priv'
        nomd = False
        if self.peek().kind in {'PUB', 'PRIV', 'PROT'}:
            access = self.bump().kind.lower()
        if self.peek().kind == 'NOMD':
            self.bump()
            nomd = True
        if self.peek().kind in TYPE_TOKENS or self.peek().kind == 'IDENT':
            typ = self.bump().value
            if self.match('STAR'):
                typ += '*'
            if self.match('LBRACKET'):
                size_tok = self.expect('INT')
                self.expect('RBRACKET')
                typ += f"[{size_tok.value}]"
        else:
            raise SyntaxError(
                f"Expected type (one of {TYPE_TOKENS} or user-defined), got {self.peek().kind} at {self.peek().line}:{self.peek().col}")
        name = self.expect('IDENT').value
        expr = None
        if self.match('EQUAL'):
            expr = self.parse_expr()
        self.expect('SEMI')
        var_decl = VarDecl(access, typ, name, expr, nomd)
        self.declared_vars[name] = var_decl
        return var_decl
    def parse_index_assign(self) -> IndexAssign:
        arr_name = self.expect('IDENT').value
        self.expect('LBRACKET')
        index = self.parse_expr()
        self.expect('RBRACKET')
        self.expect('EQUAL')
        value = self.parse_expr()
        self.expect('SEMI')
        return IndexAssign(arr_name, index, value)
    def parse_assign(self) -> Assign:
        name = self.expect('IDENT').value
        decl = self.declared_vars.get(name)
        if decl and decl.nomd:
            raise RuntimeError(f"Cannot assign to local 'nomd' variable '{name}'")
        if not decl and hasattr(self, "program"):
            for g in self.program.globals:
                if g.name == name:
                    if g.nomd:
                        raise RuntimeError(f"Cannot assign to global 'nomd' variable '{name}'")
                    break
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
    def parse_expr(self, min_prec: int = 0) -> Expr:
        left = self.parse_primary()
        while True:
            op_token = self.peek()
            if op_token.kind in {
                'PLUS', 'MINUS', 'STAR', 'SLASH', 'PERCENT',
                'EQEQ', 'NEQ', 'LT', 'LE', 'GT', 'GE',
                'AND', 'OR'
            }:
                op_prec = self.get_precedence(op_token.kind)
                if op_prec < min_prec:
                    break
                self.bump()
                right = self.parse_expr(op_prec + 1)
                left = BinOp(op_token.value, left, right)
                continue
            if op_token.kind == 'QUESTION' and min_prec == 0:
                self.bump()
                then_expr = self.parse_expr()
                self.expect('COLON')
                else_expr = self.parse_expr()
                left = Ternary(left, then_expr, else_expr)
                continue
            break
        return left
    def get_precedence(self, op: str) -> int:
        return {
            'STAR': 5, 'SLASH': 5, 'PERCENT': 5,
            'PLUS': 4, 'MINUS': 4,
            'LT': 3, 'LE': 3,
            'GT': 3, 'GE': 3,
            'EQEQ': 2, 'NEQ': 2,
            'AND': 1, 'OR': 0
        }.get(op, 0)
    def parse_primary(self) -> Expr:
        if self.peek().kind == 'STAR':
            self.bump()
            inner = self.parse_primary()
            return UnaryDeref(inner)
        if self.peek().kind == 'BANG':
            self.bump()
            inner = self.parse_primary()
            return Call("!", [inner])
        def parse_atom() -> Expr:
            t = self.bump()
            if t.kind == 'INT':
                return IntLit(int(t.value))
            if t.kind == 'FLOAT':
                val = t.value.rstrip('fF')
                return FloatLit(float(val))
            if t.kind == 'STRING':
                return StrLit(t.value)
            if t.kind == 'CHAR':
                return CharLit(t.value)
            if t.kind == 'TRUE':
                return BoolLit(True)
            if t.kind == 'FALSE':
                return BoolLit(False)
            if t.kind == 'NULL':
                return NullLit()
            if t.kind == 'LPAREN':
                expr = self.parse_expr()
                self.expect('RPAREN')
                return expr
            if t.kind == 'IDENT' and t.value == 'ORCC.get_args':
                return Call('ORCC.get_args', [])
            if t.kind == 'IDENT' and t.value in {'typeof', 'etypeof'} and self.peek().kind == 'LPAREN':
                fn = t.value
                self.bump()
                arg_expr = self.parse_expr()
                self.expect('RPAREN')
                return TypeofExpr(fn, arg_expr)
            if t.kind == 'IDENT':
                base = Var(t.value)
                if self.peek().kind == 'LBRACE':
                    self.bump()
                    fields_list: List[Tuple[str, Expr]] = []
                    while self.peek().kind != 'RBRACE':
                        fname = self.expect('IDENT').value
                        self.expect('COLON')
                        fexpr = self.parse_expr()
                        self.expect('SEMI')
                        fields_list.append((fname, fexpr))
                    self.expect('RBRACE')
                    return StructInit(t.value, fields_list)
                if self.peek().kind == 'LPAREN':
                    self.bump()
                    args: List[Expr] = []
                    if self.peek().kind != 'RPAREN':
                        while True:
                            args.append(self.parse_expr())
                            if not self.match('COMMA'):
                                break
                    self.expect('RPAREN')
                    return Call(t.value, args)
                if self.peek().kind == 'LBRACKET':
                    self.bump()
                    index_expr = self.parse_expr()
                    self.expect('RBRACKET')
                    return Index(base, index_expr)
                return base
            raise SyntaxError(f"Unexpected token: {t.kind} at {t.line}:{t.col}")
        expr: Expr = parse_atom()
        while self.peek().kind == 'DOT':
            self.bump()
            field_name = self.expect('IDENT').value
            expr = FieldAccess(expr, field_name)
        return expr
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
def unify_int_types(t1: Optional[str], t2: Optional[str]) -> Optional[str]:
    rank = {"int8": 1, "int16": 2, "int32": 3, "int64": 4, "int": 4}
    if t1 is None or t2 is None:
        return None
    for t in [t1, t2]:
        if t not in rank and not t.startswith("int"):
            return None
    return max(t1, t2, key=lambda t: rank.get(t, 0))
def unify_types(t1: str, t2: str) -> Optional[str]:
    if t1 == t2:
        return t1
    if t1 == 'null':
        return t2 if t2.endswith('*') or t2 == 'string' else None
    if t2 == 'null':
        return t1 if t1.endswith('*') or t1 == 'string' else None
    int_common = unify_int_types(t1, t2)
    if int_common:
        return int_common
    if (t1, t2) in {("float", "int"), ("int", "float")}:
        return "float"
    return None
type_map = {
    'int': 'i64', 'int8': 'i8', 'int16': 'i16', 'int32': 'i32', 'void': 'void',
    'int64': 'i64', 'float': 'double', 'bool': 'i1', 'char': 'i8', 'string': 'i8*'
}
struct_llvm_defs: List[str] = []
symbol_table = SymbolTable()
func_table: Dict[str, str] = {}
def gen_expr(expr: Expr, out: List[str]) -> str:
    def format_float(val: float) -> str:
        return f"{val:.8e}"
    global string_constants
    if isinstance(expr, IntLit):
        tmp = new_tmp()
        inferred = infer_type(expr)
        llvm_ty = type_map[inferred]
        out.append(f"  {tmp} = add {llvm_ty} 0, {expr.value}")
        return tmp
    if isinstance(expr, NullLit):
        tmp = new_tmp()
        out.append(f"  {tmp} = bitcast i8* null to i8*")
        return tmp
    if isinstance(expr, Call) and expr.name == 'ORCC.get_args':
        tmp = new_tmp()
        out.append(f"  {tmp} = load i8**, i8*** @__argv_ptr")
        return tmp
    if isinstance(expr, TypeofExpr):
        inferred = infer_type(expr.expr)
        if expr.kind == "etypeof":
            for orcat_name, llvm_name in type_map.items():
                if inferred == llvm_name:
                    inferred = orcat_name
                    break
        elif expr.kind == "typeof":
            if inferred.startswith("int") and inferred != "int":
                inferred = "int"
        else:
            raise RuntimeError(f"{expr.kind} is not a supported typeof variant")
        label = f"@.str{len(string_constants)}"
        esc = inferred.replace('"', r'\"')
        byte_len = len(inferred.encode("utf-8")) + 1
        string_constants.append(
            f'{label} = private unnamed_addr constant [{byte_len} x i8] c"{esc}\\00"'
        )
        tmp = new_tmp()
        out.append(
            f"  {tmp} = getelementptr inbounds [{byte_len} x i8], "
            f"[{byte_len} x i8]* {label}, i32 0, i32 0"
        )
        return tmp
    if isinstance(expr, FloatLit):
        tmp = new_tmp()
        float_val = format_float(expr.value)
        out.append(f"  {tmp} = fadd double 0.0, {float_val}")
        return tmp
    if isinstance(expr, BoolLit):
        tmp = new_tmp()
        val = 1 if expr.value else 0
        out.append(f"  {tmp} = add i1 0, {val}")
        return tmp
    if isinstance(expr, Ternary):
        cond_val = gen_expr(expr.cond, out)
        then_lbl = new_label("tern_then")
        else_lbl = new_label("tern_else")
        end_lbl = new_label("tern_end")
        out.append(f"  br i1 {cond_val}, label %{then_lbl}, label %{else_lbl}")
        out.append(f"{then_lbl}:")
        then_val = gen_expr(expr.then_expr, out)
        then_tmp = new_tmp()
        then_ty = type_map[infer_type(expr.then_expr)]
        out.append(f"  {then_tmp} = add {then_ty} 0, {then_val}")
        out.append(f"  br label %{end_lbl}")
        out.append(f"{else_lbl}:")
        else_val = gen_expr(expr.else_expr, out)
        else_tmp = new_tmp()
        else_ty = type_map[infer_type(expr.else_expr)]
        out.append(f"  {else_tmp} = add {else_ty} 0, {else_val}")
        out.append(f"  br label %{end_lbl}")
        out.append(f"{end_lbl}:")
        phi_tmp = new_tmp()
        out.append(f"  {phi_tmp} = phi {then_ty} [{then_tmp}, %{then_lbl}], [{else_tmp}, %{else_lbl}]")
        return phi_tmp
    if isinstance(expr, Index):
        if not isinstance(expr.array, Var):
            raise RuntimeError(f"Only direct variable array indexing is supported, got: {expr.array}")
        var_name = expr.array.name
        idx = gen_expr(expr.index, out)
        arr_info = symbol_table.lookup(var_name)
        if not arr_info:
            raise RuntimeError(f"Undefined array: {var_name}")
        llvm_ty, name = arr_info
        base_ty = extract_array_base_type(llvm_ty)
        tmp_ptr = new_tmp()
        tmp_val = new_tmp()
        idx_ty = infer_type(expr.index)
        idx_llvm = type_map[idx_ty]
        if idx_llvm != "i32":
            idx_cast = new_tmp()
            if idx_llvm.startswith("i") and int(idx_llvm[1:]) > 32:
                out.append(f"  {idx_cast} = trunc {idx_llvm} {idx} to i32")
            else:
                out.append(f"  {idx_cast} = sext {idx_llvm} {idx} to i32")
        else:
            idx_cast = idx
        len_var = f"%{var_name}_len"
        len_val = new_tmp()
        out.append(f"  {len_val} = load i32, i32* {len_var}")
        ok = new_tmp()
        out.append(f"  {ok} = icmp ult i32 {idx_cast}, {len_val}")
        fail_lbl = new_label("oob_fail")
        ok_lbl = new_label("oob_ok")
        out.append(f"  br i1 {ok}, label %{ok_lbl}, label %{fail_lbl}")
        out.append(f"{fail_lbl}:")
        out.append(f"  call void @orcc_oob_abort()")
        out.append(f"  unreachable")
        out.append(f"{ok_lbl}:")
        out.append(f"  {tmp_ptr} = getelementptr inbounds {llvm_ty}, {llvm_ty}* %{name}_addr, i32 0, i32 {idx_cast}")
        out.append(f"  {tmp_val} = load {base_ty}, {base_ty}* {tmp_ptr}")
        return tmp_val
    if isinstance(expr, StrLit):
        tmp = new_tmp()
        label = f"@.str{len(string_constants)}"
        raw = expr.value
        esc = ""
        for ch in raw:
            code = ord(ch)
            if ch == '\n':
                esc += r'\0A'
            elif ch == '\r':
                esc += r'\0D'
            elif ch == '\t':
                esc += r'\09'
            elif ch == '\\':
                esc += r'\\'
            elif ch == '"':
                esc += r'\22'
            elif 32 <= code <= 126:
                esc += ch
            else:
                esc += f'\\{code:02X}'
        byte_len = len(raw.encode('utf-8')) + 1
        string_constants.append(
            f'{label} = private unnamed_addr constant [{byte_len} x i8] c"{esc}\\00"'
        )
        out.append(
            f"  {tmp} = getelementptr inbounds [{byte_len} x i8], "
            f"[{byte_len} x i8]* {label}, i32 0, i32 0"
        )
        return tmp
    if isinstance(expr, Var):
        result = symbol_table.lookup(expr.name)
        if result is None:
            raise RuntimeError(f"Undefined variable: {expr.name}")
        typ, name = result
        tmp = new_tmp()
        if name.startswith('@'):
            out.append(f"  {tmp} = load {typ}, {typ}* {name}")
        elif name.startswith('%'):
            out.append(f"  {tmp} = load {typ}, {typ}* {name}_addr")
        else:
            out.append(f"  {tmp} = load {typ}, {typ}* %{name}_addr")
        return tmp
    if isinstance(expr, BinOp):
        lhs = gen_expr(expr.left, out)
        rhs = gen_expr(expr.right, out)
        ty = infer_type(expr.left)
        if ty == "string" and expr.op == "+":
            tmp = new_tmp()
            out.append(f"  {tmp} = call i8* @sb_append_str(i8* {lhs}, i8* {rhs})")
            return tmp
        llvm_ty = type_map[ty]
        tmp = new_tmp()
        if expr.op == '%':
            op = 'srem'
            if llvm_ty == 'double':
                op = 'frem'
            out.append(f"  {tmp} = {op} {llvm_ty} {lhs}, {rhs}")
            return tmp
        if expr.op in {'&&', '||'}:
            op = 'and' if expr.op == '&&' else 'or'
            out.append(f"  {tmp} = {op} {llvm_ty} {lhs}, {rhs}")
            return tmp
        if llvm_ty == 'double':
            op = {
                '+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv', '==': 'fcmp oeq', '!=': 'fcmp one',
                '<': 'fcmp olt', '<=': 'fcmp ole', '>': 'fcmp ogt', '>=': 'fcmp oge'
            }.get(expr.op)
        else:
            op = {
                '+': 'add',
                '-': 'sub',
                '*': 'mul',
                '/': 'sdiv',
                '%': 'srem',
                '==': 'icmp eq',
                '!=': 'icmp ne',
                '<': 'icmp slt',
                '<=': 'icmp sle',
                '>': 'icmp sgt',
                '>=': 'icmp sge',
                '&': 'and',
                '|': 'or',
                '^': 'xor',
                '<<': 'shl',
                '>>': 'ashr'
            }.get(expr.op)
        if not op:
            raise RuntimeError(f"Unsupported binary operator: {expr.op}")
        out.append(f"  {tmp} = {op} {llvm_ty} {lhs}, {rhs}")
        return tmp
    if isinstance(expr, Call):
        args_ir: List[str] = []
        arg_types: List[str] = []
        for arg in expr.args:
            a = gen_expr(arg, out)
            ty2 = infer_type(arg)
            arg_types.append(ty2)
            llvm_ty = type_map.get(ty2, f"%struct.{ty2}")
            args_ir.append(f"{llvm_ty} {a}")
        if expr.name == "!" and len(expr.args) == 1:
            arg = gen_expr(expr.args[0], out)
            arg_ty = infer_type(expr.args[0])
            if arg_ty != "bool":
                raise RuntimeError(f"Unary ! requires bool, got {arg_ty}")
            tmp = new_tmp()
            out.append(f"  {tmp} = xor i1 {arg}, true")
            return tmp
        if expr.name in func_table:
            base_fn = next((f for f in all_funcs if f.name == expr.name), None)
            if base_fn and base_fn.type_params:
                actual = arg_types[0]
                mononame = f"{expr.name}_{actual}"
                if mononame not in generated_mono:
                    subst_map = {base_fn.type_params[0]: actual}
                    new_params = [(subst_map.get(p[0], p[0]), p[1]) for p in base_fn.params]
                    new_ret = subst_map.get(base_fn.ret_type, base_fn.ret_type)
                    def replace_in_expr(e: Expr) -> Expr:
                        if isinstance(e, Var):
                            if e.name in subst_map:
                                return Var(subst_map[e.name])
                            return e
                        if isinstance(e, BinOp):
                            return BinOp(e.op, replace_in_expr(e.left), replace_in_expr(e.right))
                        if isinstance(e, Call):
                            return Call(e.name, [replace_in_expr(a0) for a0 in e.args])
                        if isinstance(e, FieldAccess):
                            return FieldAccess(replace_in_expr(e.base), e.field)
                        return e
                    def replace_in_stmt(s: Stmt) -> Stmt:
                        if isinstance(s, VarDecl):
                            typ = subst_map.get(s.typ, s.typ)
                            expr0 = replace_in_expr(s.expr) if s.expr else None
                            return VarDecl(s.access, typ, s.name, expr0)
                        if isinstance(s, Assign):
                            return Assign(s.name, replace_in_expr(s.expr))
                        if isinstance(s, IndexAssign):
                            return IndexAssign(s.array, replace_in_expr(s.index), replace_in_expr(s.value))
                        if isinstance(s, IfStmt):
                            cond0 = replace_in_expr(s.cond)
                            then_body0 = [replace_in_stmt(ss) for ss in s.then_body]
                            else_body0 = None
                            if s.else_body:
                                if isinstance(s.else_body, IfStmt):
                                    else_body0 = replace_in_stmt(s.else_body)
                                else:
                                    else_body_list: List[Stmt] = []
                                    for ss in s.else_body:
                                        else_body_list.extend([replace_in_stmt(ss)])
                                    else_body0 = else_body_list
                            return IfStmt(cond0, then_body0, else_body0)
                        if isinstance(s, WhileStmt):
                            cond0 = replace_in_expr(s.cond)
                            body0 = [replace_in_stmt(ss) for ss in s.body]
                            return WhileStmt(cond0, body0)
                        if isinstance(s, ReturnStmt):
                            return ReturnStmt(replace_in_expr(s.expr) if s.expr else None)
                        if isinstance(s, ExprStmt):
                            return ExprStmt(replace_in_expr(s.expr))
                        if isinstance(s, Match):
                            new_expr0 = replace_in_expr(s.expr)
                            new_cases: List[MatchCase] = []
                            for case in s.cases:
                                new_body0 = [replace_in_stmt(ss) for ss in case.body]
                                new_cases.append(MatchCase(case.variant, case.binding, new_body0))
                            return Match(new_expr0, new_cases)
                        raise RuntimeError(f"Unsupported Stmt in generic substitution: {s}")
                    new_body = [replace_in_stmt(stmt) for stmt in base_fn.body] if base_fn.body else None
                    new_fn = Func(base_fn.access, mononame, [], new_params, new_ret, new_body, base_fn.is_extern)
                    all_funcs.append(new_fn)
                    func_table[mononame] = type_map.get(new_ret, f"%struct.{new_ret}")
                    llvm_lines = gen_func(new_fn)
                    out.insert(0, "\n".join(llvm_lines))
                ret_ty = type_map.get(actual, f"%struct.{actual}")
                tmp2 = new_tmp()
                out.append(f"  {tmp2} = call {ret_ty} @{mononame}({', '.join(args_ir)})")
                return tmp2
        ret_ty = func_table.get(expr.name, 'i32')
        if ret_ty == 'void':
            out.append(f"  call void @{expr.name}({', '.join(args_ir)})")
            return ''
        else:
            tmp2 = new_tmp()
            out.append(f"  {tmp2} = call {ret_ty} @{expr.name}({', '.join(args_ir)})")
            return tmp2
    if isinstance(expr, FieldAccess):
        base_ptr = gen_expr(expr.base, out)
        base_type = infer_type(expr.base).rstrip("*").removeprefix("%struct.")
        if base_type not in struct_field_map:
            raise RuntimeError(f"Struct type '{base_type}' not found")
        fields = struct_field_map[base_type]
        field_dict = dict(fields)
        if expr.field not in field_dict:
            raise RuntimeError(f"Struct '{base_type}' has no field '{expr.field}'")
        index = list(field_dict.keys()).index(expr.field)
        field_typ = field_dict[expr.field]
        field_llvm = llvm_type_of(field_typ)
        ptr = new_tmp()
        out.append(f"  {ptr} = getelementptr inbounds %struct.{base_type}, %struct.{base_type}* {base_ptr}, i32 0, i32 {index}")
        tmp = new_tmp()
        out.append(f"  {tmp} = load {field_llvm}, {field_llvm}* {ptr}")
        return tmp
    if isinstance(expr, StructInit):
        struct_name = expr.name
        struct_ty = f"%struct.{struct_name}"
        tmp_ptr = new_tmp()
        out.append(f"  {tmp_ptr} = alloca {struct_ty}")
        field_dict = dict(struct_field_map[struct_name])
        for field_name, field_expr in expr.fields:
            if field_name not in field_dict:
                raise RuntimeError(f"Field '{field_name}' not in struct '{struct_name}'")
            field_type = field_dict[field_name]
            field_llvm = llvm_type_of(field_type)
            field_val = gen_expr(field_expr, out)
            index = list(field_dict.keys()).index(field_name)
            ptr = new_tmp()
            out.append(f"  {ptr} = getelementptr inbounds {struct_ty}, {struct_ty}* {tmp_ptr}, i32 0, i32 {index}")
            out.append(f"  store {field_llvm} {field_val}, {field_llvm}* {ptr}")
        return tmp_ptr
    raise RuntimeError(f"Unhandled expr: {expr}")
def infer_type(expr: Expr) -> str:
    if isinstance(expr, UnaryDeref):
        ptr_type = infer_type(expr.ptr)
        if not ptr_type.endswith("*"):
            raise RuntimeError(f"Dereferencing non-pointer type '{ptr_type}'")
        return ptr_type[:-1]
    if isinstance(expr, IntLit):
        return 'int'
    if isinstance(expr, FloatLit):
        return 'float'
    if isinstance(expr, BoolLit):
        return 'bool'
    if isinstance(expr, CharLit):
        return 'char'
    if isinstance(expr, StrLit):
        return 'string'
    if isinstance(expr, NullLit):
        return 'void*'
    if isinstance(expr, Var):
        result = symbol_table.lookup(expr.name)
        if result is None:
            raise RuntimeError(f"Undefined variable: {expr.name}")
        llvm_ty, _ = result
        if llvm_ty.startswith('[') and ' x ' in llvm_ty and llvm_ty.endswith(']'):
            inside = llvm_ty[1:-1]
            elem_llvm = inside.split(' x ')[1]
            for high, low in type_map.items():
                if low == elem_llvm:
                    return high
            return elem_llvm
        for high, low in type_map.items():
            if low == llvm_ty:
                return high
        if llvm_ty.startswith("%struct.") and llvm_ty.endswith("*"):
            return llvm_ty[len("%struct."):-1] + "*"
        for high, low in type_map.items():
            if low == llvm_ty:
                return high
        return llvm_ty
    if isinstance(expr, TypeofExpr):
        if expr.kind in {'typeof', 'etypeof'}:
            return 'string'
    if isinstance(expr, FieldAccess):
        base_type = clean_struct_name(infer_type(expr.base))
        if base_type not in struct_field_map:
            raise RuntimeError(f"Struct type '{base_type}' not found")
        fields = struct_field_map[base_type]
        field_dict = dict(fields)
        if expr.field not in field_dict:
            raise RuntimeError(f"Field '{expr.field}' not in struct '{base_type}'")
        return field_dict[expr.field]
    if isinstance(expr, StructInit):
        return expr.name + "*"
    if isinstance(expr, BinOp):
        left_type = infer_type(expr.left)
        right_type = infer_type(expr.right)
        common = unify_int_types(left_type, right_type)
        if not common:
            if left_type != right_type:
                raise RuntimeError(f"Type mismatch in binary op '{expr.op}': {left_type} vs {right_type}")
            common = left_type
        if expr.op in {'==', '!=', '<', '<=', '>', '>='}:
            return 'bool'
        return common
    if isinstance(expr, Call):
        if expr.name == "!" and len(expr.args) == 1:
            if infer_type(expr.args[0]) != "bool":
                raise RuntimeError("Unary ! requires bool operand")
            return "bool"
        if expr.name == "exit":
            return "void"
        if expr.name not in func_table:
            raise RuntimeError(f"Call to undefined function '{expr.name}'")
        ret_llvm_ty = func_table[expr.name]
        for high, low in type_map.items():
            if low == ret_llvm_ty:
                return high
        if ret_llvm_ty.startswith("%struct.") and ret_llvm_ty.endswith("*"):
            return ret_llvm_ty[len("%struct."):-1] + "*"
        return ret_llvm_ty
    if isinstance(expr, Index):
        if not isinstance(expr.array, Var):
            raise RuntimeError(f"Only direct variable array indexing is supported, got: {expr.array}")
        arr_name = expr.array.name
        arr_info = symbol_table.lookup(arr_name)
        if not arr_info:
            raise RuntimeError(f"Undefined array: {arr_name}")
        llvm_ty, _ = arr_info
        if not (llvm_ty.startswith('[') and ' x ' in llvm_ty and llvm_ty.endswith(']')):
            raise RuntimeError(f"Attempting to index non-array type '{llvm_ty}'")
        inside = llvm_ty[1:-1]
        elem_llvm = inside.split(' x ')[1]
        for high, low in type_map.items():
            if low == elem_llvm:
                return high
        return elem_llvm
    if isinstance(expr, Ternary):
        then_t = infer_type(expr.then_expr)
        else_t = infer_type(expr.else_expr)
        if then_t != else_t:
            raise RuntimeError(f"Ternary branches must match: {then_t} vs {else_t}")
        return then_t
    if hasattr(expr, '__dict__'):
        possible = expr.__dict__.get("name", "")
        if isinstance(possible, str) and possible.endswith("*"):
            return possible
    raise RuntimeError(f"Cannot infer type for expression: {expr}")
def gen_stmt(stmt: Stmt, out: List[str], ret_ty: str):
    if isinstance(stmt, VarDecl):
        if "[" in stmt.typ:
            base, count = stmt.typ.split("[")
            count = count[:-1]
            llvm_ty = f"[{count} x {type_map[base]}]"
            if not symbol_table.lookup(stmt.name):
                out.append(f"  %{stmt.name}_addr = alloca {llvm_ty}")
                out.append(f"  %{stmt.name}_len  = alloca i32")
                out.append(f"  store i32 {count}, i32* %{stmt.name}_len")
                symbol_table.declare(stmt.name, llvm_ty, stmt.name)
        else:
            if stmt.typ.endswith("*"):
                base = stmt.typ.rstrip("*")
                base_llvm = type_map.get(base, f"%struct.{base}")
                llvm_ty = f"{base_llvm}*"
            else:
                llvm_ty = type_map.get(stmt.typ, f"%struct.{stmt.typ}")
            if not symbol_table.lookup(stmt.name):
                out.append(f"  %{stmt.name}_addr = alloca {llvm_ty}")
                symbol_table.declare(stmt.name, llvm_ty, stmt.name)
        if stmt.expr:
            val = gen_expr(stmt.expr, out)
            src_llvm = type_map.get(infer_type(stmt.expr), f"%struct.{infer_type(stmt.expr)}")
            if src_llvm != llvm_ty:
                cast_tmp = new_tmp()
                bits_src = llvm_int_bitsize(src_llvm)
                bits_dst = llvm_int_bitsize(llvm_ty)
                if bits_src and bits_dst:
                    if bits_src > bits_dst:
                        out.append(f"  {cast_tmp} = trunc {src_llvm} {val} to {llvm_ty}")
                    else:
                        out.append(f"  {cast_tmp} = sext {src_llvm} {val} to {llvm_ty}")
                    val = cast_tmp
            out.append(f"  store {llvm_ty} {val}, {llvm_ty}* %{stmt.name}_addr")
        return
    elif isinstance(stmt, Assign):
        if isinstance(stmt.name, UnaryDeref):
            ptr_val = gen_expr(stmt.name.ptr, out)
            val = gen_expr(stmt.expr, out)
            val_ty = infer_type(stmt.expr)
            llvm_ty = type_map.get(val_ty, f"%struct.{val_ty}")
            out.append(f"  store {llvm_ty} {val}, {llvm_ty}* {ptr_val}")
        else:
            val = gen_expr(stmt.expr, out)
            llvm_ty, ir_name = symbol_table.lookup(stmt.name)
            if ir_name.startswith('@'):
                out.append(f"  store {llvm_ty} {val}, {llvm_ty}* {ir_name}")
            else:
                out.append(f"  store {llvm_ty} {val}, {llvm_ty}* %{ir_name}_addr")
    elif isinstance(stmt, ContinueStmt):
        if not loop_stack:
            raise RuntimeError("`continue` used outside of a loop")
        head_lbl = loop_stack[-1]['continue']
        out.append(f"  br label %{head_lbl}")
        out.append("  unreachable")
    elif isinstance(stmt, BreakStmt):
        if not loop_stack:
            raise RuntimeError("`break` used outside of a loop")
        break_lbl = loop_stack[-1]['break']
        out.append(f"  br label %{break_lbl}")
        out.append("  unreachable")
    elif isinstance(stmt, IndexAssign):
        idx = gen_expr(stmt.index, out)
        val = gen_expr(stmt.value, out)
        llvm_ty, name = symbol_table.lookup(stmt.array)
        base_ty = extract_array_base_type(llvm_ty)
        idx_ty = infer_type(stmt.index)
        idx_llvm = type_map[idx_ty]
        if idx_llvm != "i32":
            idx_cast = new_tmp()
            if idx_llvm.startswith("i") and int(idx_llvm[1:]) > 32:
                out.append(f"  {idx_cast} = trunc {idx_llvm} {idx} to i32")
            else:
                out.append(f"  {idx_cast} = sext {idx_llvm} {idx} to i32")
        else:
            idx_cast = idx
        len_var = f"%{stmt.array}_len"
        len_val = new_tmp()
        out.append(f"  {len_val} = load i32, i32* {len_var}")
        ok = new_tmp()
        out.append(f"  {ok} = icmp ult i32 {idx_cast}, {len_val}")
        fail_lbl = new_label("oob_fail")
        ok_lbl   = new_label("oob_ok")
        out.append(f"  br i1 {ok}, label %{ok_lbl}, label %{fail_lbl}")
        out.append(f"{fail_lbl}:")
        out.append(f"  call void @orcc_oob_abort()")
        out.append(f"  unreachable")
        out.append(f"{ok_lbl}:")
        ptr_tmp = new_tmp()
        out.append(f"  {ptr_tmp} = getelementptr inbounds {llvm_ty}, {llvm_ty}* %{name}_addr, i32 0, i32 {idx_cast}")
        out.append(f"  store {base_ty} {val}, {base_ty}* {ptr_tmp}")
    elif isinstance(stmt, IfStmt):
        cond = gen_expr(stmt.cond, out)
        then_lbl = new_label('then')
        else_lbl = new_label('else') if stmt.else_body else None
        end_lbl = new_label('endif')
        out.append(f"  br i1 {cond}, label %{then_lbl}, label %{else_lbl or end_lbl}")
        out.append(f"{then_lbl}:")
        symbol_table.push()
        for s in stmt.then_body:
            gen_stmt(s, out, ret_ty)
        symbol_table.pop()
        out.append(f"  br label %{end_lbl}")
        if stmt.else_body:
            out.append(f"{else_lbl}:")
            symbol_table.push()
            if isinstance(stmt.else_body, list):
                for s in stmt.else_body:
                    gen_stmt(s, out, ret_ty)
            elif isinstance(stmt.else_body, IfStmt):
                gen_stmt(stmt.else_body, out, ret_ty)
            symbol_table.pop()
            out.append(f"  br label %{end_lbl}")
        out.append(f"{end_lbl}:")
    elif isinstance(stmt, WhileStmt):
        head_lbl = new_label('while_head')
        body_lbl = new_label('while_body')
        end_lbl = new_label('while_end')
        loop_stack.append({'continue': head_lbl, 'break': end_lbl})
        out.append(f"  br label %{head_lbl}")
        out.append(f"{head_lbl}:")
        cond = gen_expr(stmt.cond, out)
        out.append(f"  br i1 {cond}, label %{body_lbl}, label %{end_lbl}")
        out.append(f"{body_lbl}:")
        symbol_table.push()
        for s in stmt.body:
            gen_stmt(s, out, ret_ty)
        symbol_table.pop()
        out.append(f"  br label %{head_lbl}")
        out.append(f"{end_lbl}:")
        loop_stack.pop()
    elif isinstance(stmt, ReturnStmt):
        val = gen_expr(stmt.expr, out) if stmt.expr else None
        if val:
            ret_type = infer_type(stmt.expr)
            llvm_ty = type_map.get(ret_type, f"%struct.{ret_type}")
            if ret_ty == "i32" and llvm_ty != "i32":
                tmp = new_tmp()
                if llvm_ty.startswith("i") and llvm_ty[1:].isdigit():
                    bits = int(llvm_ty[1:])
                    if bits > 32:
                        out.append(f"  {tmp} = trunc {llvm_ty} {val} to i32")
                    else:
                        out.append(f"  {tmp} = sext {llvm_ty} {val} to i32")
                    out.append(f"  ret i32 {tmp}")
                else:
                    out.append(f"  ret i32 0")
            else:
                out.append(f"  ret {llvm_ty} {val}")
        else:
            out.append(f"  ret void")
    elif isinstance(stmt, ExprStmt):
        gen_expr(stmt.expr, out)
    elif isinstance(stmt, DerefStmt):
        llvm_ty, llvm_name = symbol_table.lookup(stmt.varname)
        ptr_tmp = new_tmp()
        out.append(f"  {ptr_tmp} = load {llvm_ty}, {llvm_ty}* %{llvm_name}_addr")
        if not llvm_ty.endswith("*"):
            raise RuntimeError(f"Cannot deref non-pointer type '{llvm_ty}'")
        cast_tmp = new_tmp()
        out.append(f"  {cast_tmp} = bitcast {llvm_ty} {ptr_tmp} to i8*")
        out.append(f"  call void @free(i8* {cast_tmp})")
    elif isinstance(stmt, Match):
        enum_ptr = gen_expr(stmt.expr, out)
        tag_tmp = new_tmp()
        out.append(f"  {tag_tmp} = getelementptr inbounds %enum.{infer_type(stmt.expr)}, %enum.{infer_type(stmt.expr)}* {enum_ptr}, i32 0, i32 0")
        loaded_tag = new_tmp()
        out.append(f"  {loaded_tag} = load i32, i32* {tag_tmp}")
        end_lbl = new_label("match_end")
        variant_labels = {}
        for case in stmt.cases:
            lbl = new_label(f"case_{case.variant}")
            variant_labels[case.variant] = lbl
        out.append(f"  switch i32 {loaded_tag}, label %{end_lbl} [")
        for idx, (vname, payload) in enumerate(enum_variant_map[infer_type(stmt.expr)]):
            lbl = variant_labels[vname]
            out.append(f"    i32 {idx}, label %{lbl}")
        out.append("  ]")
        for idx, case in enumerate(stmt.cases):
            lbl = variant_labels[case.variant]
            out.append(f"{lbl}:")
            variant_info = next(v for v in enum_variant_map[infer_type(stmt.expr)] if v[0] == case.variant)
            payload_type = variant_info[1]
            if payload_type is not None:
                payload_ptr = new_tmp()
                out.append(f"  {payload_ptr} = getelementptr inbounds %enum.{infer_type(stmt.expr)}, %enum.{infer_type(stmt.expr)}* {enum_ptr}, i32 0, i32 1")
                loaded_payload = new_tmp()
                llvm_payload_ty = type_map.get(payload_type, f"%struct.{payload_type}")
                out.append(f"  {loaded_payload} = load {llvm_payload_ty}, {llvm_payload_ty}* {payload_ptr}")
                var_name = case.binding
                if var_name is not None:
                    out.append(f"  %{var_name}_addr = alloca {llvm_payload_ty}")
                    out.append(f"  store {llvm_payload_ty} {loaded_payload}, {llvm_payload_ty}* %{var_name}_addr")
                    symbol_table.declare(var_name, llvm_payload_ty, var_name)
            for s in case.body:
                gen_stmt(s, out, ret_ty)
            out.append(f"  br label %{end_lbl}")
        out.append(f"{end_lbl}:")
def gen_func(fn: Func) -> List[str]:
    if fn.type_params:
        return []
    def llvm_ty_of(typ: str) -> str:
        base = typ.rstrip('*')
        if base in type_map:
            return f"{type_map[base]}*" if typ.endswith("*") else type_map[base]
        struct_ir = f"%struct.{base}"
        return f"{struct_ir}*" if typ.endswith("*") else struct_ir
    if fn.is_extern:
        param_sig = ", ".join(f"{llvm_ty_of(t)} %{n}" for t, n in fn.params)
        ret_ty    = llvm_ty_of(fn.ret_type)
        generated_mono[fn.name] = True
        return [f"declare {ret_ty} @{fn.name}({param_sig})"]
    symbol_table.push()
    generated_mono[fn.name] = True
    if fn.name == "main":
        ret_ty = "i32"
        out    = [f"define i32 @main(i32 %argc, i8** %argv) {{", "entry:"]
        out.append("  store i8** %argv, i8*** @__argv_ptr")
    else:
        ret_ty    = llvm_ty_of(fn.ret_type)
        param_sig = ", ".join(f"{llvm_ty_of(t)} %{n}" for t, n in fn.params)
        out       = [f"define {ret_ty} @{fn.name}({param_sig}) {{", "entry:"]
    for typ, name in fn.params:
        llvm_ty = llvm_ty_of(typ)
        out.append(f"  %{name}_addr = alloca {llvm_ty}")
        out.append(f"  store {llvm_ty} %{name}, {llvm_ty}* %{name}_addr")
        symbol_table.declare(name, llvm_ty, name)
    has_return = False
    for stmt in fn.body or []:
        if isinstance(stmt, ReturnStmt):
            gen_stmt(stmt, out, ret_ty)
            has_return = True
            break
        else:
            gen_stmt(stmt, out, ret_ty)
    for reg in extension_registry["registrations"].values():
        if reg.get("type") == "function" and fn.name.startswith("__launch_"):
            for line in reg.get("actions", []):
                if "<selfname>" in line:
                    line = line.replace("<selfname>", fn.name[len("__launch_"):])
                out.append(f"  {line}")
    if not has_return:
        if ret_ty == 'void':
            out.append("  ret void")
        elif ret_ty == 'double':
            out.append(f"  ret {ret_ty} 0.0")
        elif ret_ty == 'i8*':
            out.append("  ret i8* null")
        elif ret_ty.startswith('i'):
            out.append(f"  ret {ret_ty} 0")
        else:
            out.append(f"  ret {ret_ty} null")
    out.append("}")
    symbol_table.pop()
    return out
def compile_program(prog: Program) -> str:
    global all_funcs, func_table, builtins_emitted
    all_funcs = prog.funcs[:]
    string_constants.clear()
    func_table.clear()
    for fn in prog.funcs:
        if fn.type_params:
            continue
        llvm_ret_ty = type_map.get(fn.ret_type, f"%struct.{fn.ret_type}")
        func_table[fn.name] = llvm_ret_ty
    func_table["exit"] = "void"
    func_table["malloc"] = "i8*"
    func_table["free"] = "void"
    has_user_main = False
    for fn in prog.funcs:
        if fn.name == "main":
            has_user_main = True
            fn.name = "user_main"
            llvm_ret_ty = type_map.get(fn.ret_type, f"%struct.{fn.ret_type}")
            func_table.pop("main", None)
            func_table["user_main"] = llvm_ret_ty
            break
    lines: List[str] = [
        "; ModuleID = 'orcat'",
        f"source_filename = \"{compiled}\"",
        "@__argv_ptr = global i8** null",
        "declare i8* @malloc(i64)",
        "declare void @free(i8*)",
        "",
        "declare void @puts(i8*)",
        "declare void @exit(i64)",
        """
    @.oob_msg = private unnamed_addr constant [52 x i8] c"[ORCatCompiler-RT-CHCK]: Index out of bounds error.\\00"
    define void @orcc_oob_abort() {
    entry:
      call void @puts(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.oob_msg, i32 0, i32 0))
      call void @exit(i64 1)
      unreachable
    }
    """
    ]
    for global_line in extension_registry["llvm.globals"]:
        lines.append(global_line)
    for g in prog.globals:
        if "[" in g.typ and g.typ.endswith("]"):
            base, count = g.typ.split("[")
            count = count[:-1]
            base_llvm = type_map.get(base, f"%struct.{base}")
            llvm_ty = f"[{count} x {base_llvm}]"
        else:
            llvm_ty = type_map.get(g.typ, f"%struct.{g.typ}")
        initializer = "zeroinitializer"
        if isinstance(g.expr, IntLit):
            initializer = str(g.expr.value)
        elif isinstance(g.expr, FloatLit):
            initializer = f"{g.expr.value:.8e}"
        elif isinstance(g.expr, BoolLit):
            initializer = "1" if g.expr.value else "0"
        elif isinstance(g.expr, CharLit):
            initializer = str(ord(g.expr.value))
        elif isinstance(g.expr, StrLit):
            label = f"@.str{len(string_constants)}"
            esc = ""
            for ch in g.expr.value:
                if ch == '\n': esc += r'\0A'
                elif ch == '\t': esc += r'\09'
                elif ch == '\\': esc += r'\\'
                elif ch == '"': esc += r'\"'
                else: esc += ch
            length = len(g.expr.value) + 1
            string_constants.append(
                f'{label} = private unnamed_addr constant [{length} x i8] c"{esc}\\00"')
            initializer = f"getelementptr inbounds ([{length} x i8], [{length} x i8]* {label}, i32 0, i32 0)"
            llvm_ty = "i8*"
        if initializer.startswith("getelementptr"):
            lines.append(f"@{g.name} = global {llvm_ty} {initializer}")
        else:
            lines.append(f"@{g.name} = global {llvm_ty} {initializer}")
        symbol_table.declare(g.name, llvm_ty, f"@{g.name}")
    struct_llvm_defs: List[str] = []
    for sdef in prog.structs:
        field_tys: List[str] = []
        struct_field_map[sdef.name] = [(f.name, f.typ) for f in sdef.fields]
        for fld in sdef.fields:
            if fld.typ in type_map:
                field_tys.append(type_map[fld.typ])
            else:
                field_tys.append(f"%struct.{fld.typ}")
        llvm_line = f"%struct.{sdef.name} = type {{ {', '.join(field_tys)} }}"
        struct_llvm_defs.append(llvm_line)
    if struct_llvm_defs:
        lines.extend(struct_llvm_defs)
        lines.append("")
    for ename, variants in enum_variant_map.items():
        llvm_line = f"%enum.{ename} = type {{ i32, [8 x i8] }}"
        lines.append(llvm_line)
    if enum_variant_map:
        lines.append("")
    existing_globals = set()
    existing_funcs = set()
    for line in lines:
        m_g = re.match(r'\s*@(\w+)\s*=', line)
        if m_g:
            existing_globals.add(m_g.group(1))
        m_f = re.match(r'\s*define\s+[^(]+\s+@(\w+)\s*\(', line)
        if m_f:
            existing_funcs.add(m_f.group(1))
    if not builtins_emitted and "orcat_argc_global" not in existing_globals:
        func_table["orcat_argc"] = "i64"
        func_table["orcat_argv_i"] = "i8*"
        lines.append("@orcat_argc_global = global i64 0")
        lines.append("@orcat_argv_global = global i8** null")
        lines.append("")
        lines.append("")
        builtins_emitted = True
    for fn in prog.funcs:
        lines += gen_func(fn)
    if string_constants:
        lines.extend(string_constants)
        lines.append("")
        string_constants.clear()
    if has_user_main:
        lines.append("define i32 @main(i32 %argc, i8** %argv) {")
        lines.append("entry:")
        lines.append("  %argc64 = sext i32 %argc to i64")
        lines.append("  store i64 %argc64, i64* @orcat_argc_global")
        lines.append("  store i8** %argv, i8*** @orcat_argv_global")
        lines.append("  %ret64 = call i64 @user_main()")
        lines.append("  %ret32 = trunc i64 %ret64 to i32")
        lines.append("  ret i32 %ret32")
        lines.append("}")
    return "\n".join(lines)
def check_types(prog: Program):
    env = TypeEnv()
    crumb_map: Dict[str, Tuple[Optional[int], Optional[int], int, int]] = {}
    funcs = {f.name: f for f in prog.funcs}
    for g in prog.globals:
        env.declare(g.name, g.typ)
        if g.nomd:
            crumb_map[g.name] = (None, 0, 0, 0)
    struct_defs: Dict[str, StructDef] = {s.name: s for s in prog.structs}
    enum_defs:   Dict[str, EnumDef]   = {e.name: e for e in prog.enums}
    for sdef in prog.structs:
        struct_field_map[sdef.name] = [(fld.name, fld.typ) for fld in sdef.fields]
    for struct_name in struct_defs:
        env.declare(struct_name, struct_name)
    for enum_name in enum_defs:
        env.declare(enum_name, enum_name)
    def check_expr(expr: Expr) -> str:
        if isinstance(expr, IntLit):
            return 'int'
        if isinstance(expr, FloatLit):
            return 'float'
        if isinstance(expr, BoolLit):
            return 'bool'
        if isinstance(expr, CharLit):
            return 'char'
        if isinstance(expr, StrLit):
            return 'string'
        if isinstance(expr, NullLit):
            return 'null'
        if isinstance(expr, Var):
            typ = env.lookup(expr.name)
            if not typ:
                raise TypeError(f"Use of undeclared variable '{expr.name}'")
            if typ == "undefined":
                raise TypeError(f"Use of variable '{expr.name}' after deref (use-after-free)")
            if expr.name in crumb_map:
                rmax, wmax, rc, wc = crumb_map[expr.name]
                crumb_map[expr.name] = (rmax, wmax, rc + 1, wc)
            return typ
        if isinstance(expr, TypeofExpr):
            inner_type = check_expr(expr.expr)
            if expr.kind in {"typeof", "etypeof"}:
                return "string"
            elif expr.kind in {"kwtypeof", "kwetypeof"}:
                return inner_type
        if isinstance(expr, Ternary):
            cond_type = check_expr(expr.cond)
            if cond_type != 'bool':
                raise TypeError("Ternary condition must be bool")
            then_t = check_expr(expr.then_expr)
            else_t = check_expr(expr.else_expr)
            if then_t != else_t:
                raise TypeError(f"Ternary branches must match: {then_t} vs {else_t}")
            return then_t
        if isinstance(expr, BinOp):
            left = check_expr(expr.left)
            right = check_expr(expr.right)
            if expr.op == "+" and left == "string" and right == "string":
                return "string"
            if expr.op in {"&&", "||"}:
                if left != "bool" or right != "bool":
                    raise TypeError(
                        f"Logical '{expr.op}' requires both operands to be bool, got {left} and {right}")
                return "bool"
            if expr.op == "%":
                if left == "int" and right == "int":
                    return "int"
                elif left == "float" and right == "float":
                    return "float"
                else:
                    raise TypeError(f"Modulo '%' requires int or float, got {left} and {right}")
            common = unify_types(left, right)
            if not common:
                raise TypeError(f"Type mismatch: {left} {expr.op} {right}")
            if expr.op in {"==", "!=", "<", ">", "<=", ">="}:
                return "bool"
            return common
        if isinstance(expr, Call):
            if expr.name == "!" and len(expr.args) == 1:
                arg_type = check_expr(expr.args[0])
                if arg_type != "bool":
                    raise TypeError("Unary ! requires a bool")
                return "bool"
            fn = funcs.get(expr.name)
            if fn is None:
                if expr.name == "exit":
                    if len(expr.args) != 1:
                        raise TypeError("exit() takes exactly one int argument")
                    arg_ty = check_expr(expr.args[0])
                    if not arg_ty.startswith("int"):
                        raise TypeError("exit() expects an integer argument")
                    return "void"
                if expr.name == "malloc":
                    if len(expr.args) != 1:
                        raise TypeError("malloc() takes exactly one int argument")
                    arg_ty = check_expr(expr.args[0])
                    if not arg_ty.startswith("int"):
                        raise TypeError("malloc() expects an integer argument")
                    return "int*"
                if expr.name == "free":
                    if len(expr.args) != 1:
                        raise TypeError("free() takes exactly one pointer argument")
                    arg_ty = check_expr(expr.args[0])
                    if not arg_ty.endswith("*"):
                        raise TypeError("free() expects a pointer argument")
                    return "void"
                raise TypeError(f"Call to undeclared function '{expr.name}'")
            if fn.type_params:
                if len(fn.type_params) != 1:
                    raise TypeError(f"Only single-type-param generics supported, but got {fn.type_params}")
                type_arg = check_expr(expr.args[0])
                type_param = fn.type_params[0]
                mononame = f"{expr.name}_{type_arg}"
                if mononame not in funcs:
                    new_params = [(type_arg if t == type_param else t, name) for (t, name) in fn.params]
                    new_ret = type_arg if fn.ret_type == type_param else fn.ret_type
                    def substitute(e: Expr) -> Expr:
                        if isinstance(e, Var) and e.name == type_param:
                            return Var(type_arg)
                        if isinstance(e, BinOp):
                            return BinOp(e.op, substitute(e.left), substitute(e.right))
                        if isinstance(e, Call):
                            return Call(e.name, [substitute(arg) for arg in e.args])
                        return e
                    def substitute_stmt(s: Stmt) -> Stmt:
                        if isinstance(s, VarDecl):
                            typ = type_arg if s.typ == type_param else s.typ
                            expr2 = substitute(s.expr) if s.expr else None
                            return VarDecl(s.access, typ, s.name, expr2)
                        if isinstance(s, Assign):
                            return Assign(s.name, substitute(s.expr))
                        if isinstance(s, ReturnStmt):
                            return ReturnStmt(substitute(s.expr) if s.expr else None)
                        if isinstance(s, ExprStmt):
                            return ExprStmt(substitute(s.expr))
                        return s
                    new_body = [substitute_stmt(s) for s in fn.body] if fn.body else None
                    new_fn = Func(fn.access, mononame, [], new_params, new_ret, new_body, fn.is_extern)
                    funcs[mononame] = new_fn
                    prog.funcs.append(new_fn)
                expr.name = mononame
                fn = funcs[mononame]
            if not fn.is_extern:
                if len(expr.args) != len(fn.params):
                    raise TypeError(f"Arity mismatch in call to '{expr.name}'")
                for arg_expr, (expected_type, _) in zip(expr.args, fn.params):
                    actual_type = check_expr(arg_expr)
                    common = unify_int_types(actual_type, expected_type)
                    if expected_type != "void" and actual_type != expected_type and (not common or common != expected_type):
                        raise TypeError(
                            f"Argument type mismatch in call to '{expr.name}': "
                            f"expected {expected_type}, got {actual_type}")
            return fn.ret_type
        if isinstance(expr, Index):
            var_typ = env.lookup(expr.array.name)
            if not var_typ:
                raise TypeError(f"Indexing undeclared variable '{expr.array.name}'")
            if '[' not in var_typ or not var_typ.endswith(']'):
                raise TypeError(f"Attempting to index non-array type '{var_typ}'")
            base_type = var_typ.split('[', 1)[0]
            idx_type = check_expr(expr.index)
            if idx_type != 'int':
                raise TypeError(f"Array index must be 'int', got '{idx_type}'")
            return base_type
        if isinstance(expr, FieldAccess):
            base_type = check_expr(expr.base)
            if base_type not in struct_defs:
                raise TypeError(f"Attempting field access on non‐struct type '{base_type}'")
            fields = struct_field_map[base_type]
            for (fname, ftyp) in fields:
                if fname == expr.field:
                    return ftyp
            raise TypeError(f"Struct '{base_type}' has no field '{expr.field}'")
        if isinstance(expr, StructInit):
            if expr.name not in struct_defs:
                raise TypeError(f"Unknown struct type '{expr.name}' in initializer")
            expected_fields = struct_field_map[expr.name][:]
            seen_fields = set()
            for (fname, fexpr) in expr.fields:
                match_list = [ft for (fn, ft) in expected_fields if fn == fname]
                if not match_list:
                    raise TypeError(f"Struct '{expr.name}' has no field '{fname}'")
                declared_type = match_list[0]
                actual_type = check_expr(fexpr)
                if actual_type != declared_type:
                    raise TypeError(
                        f"Struct '{expr.name}' field '{fname}': expected '{declared_type}', got '{actual_type}'")
                seen_fields.add(fname)
            all_field_names = {fn for (fn, _) in expected_fields}
            if seen_fields != all_field_names:
                missing = all_field_names - seen_fields
                raise TypeError(f"Struct '{expr.name}' initializer missing fields {missing}")
            return expr.name + "*"
    def check_stmt(stmt: Stmt, expected_ret: str):
        if isinstance(stmt, VarDecl):
            if env.lookup(stmt.name):
                raise TypeError(f"Variable '{stmt.name}' already declared")
            raw_typ = stmt.typ
            base_type = raw_typ.rstrip('*')
            if '[' in base_type and base_type.endswith(']'):
                base_type = base_type.split('[', 1)[0]
            if base_type not in type_map and base_type not in struct_defs and base_type not in enum_defs:
                raise TypeError(f"Unknown type '{raw_typ}'")
            env.declare(stmt.name, raw_typ)
            if stmt.expr:
                expr_type = check_expr(stmt.expr)
                if stmt.name in crumb_map:
                    rmax, wmax, rc, wc = crumb_map[stmt.name]
                    crumb_map[stmt.name] = (rmax, wmax, rc, wc + 1)
                if isinstance(stmt.expr, IntLit):
                    int_targets = {"int8", "int16", "int32", "int64"}
                    if raw_typ in int_targets:
                        return
                common = unify_types(expr_type, raw_typ)
                if expr_type != raw_typ and (not common or common != raw_typ):
                    raise TypeError(
                        f"Type mismatch in variable init '{stmt.name}': expected {raw_typ}, got {expr_type}")
        elif isinstance(stmt, ContinueStmt):
            return
        elif isinstance(stmt, BreakStmt):
            return
        elif isinstance(stmt, Assign):
            if isinstance(stmt.name, UnaryDeref):
                ptr_type = check_expr(stmt.name.ptr)
                if not ptr_type.endswith('*'):
                    raise TypeError(f"Dereferencing non-pointer type '{ptr_type}'")
                pointee = ptr_type[:-1]
                expr_type = check_expr(stmt.expr)
                if expr_type != pointee:
                    raise TypeError(
                        f"Pointer-assign type mismatch: attempted to store '{expr_type}' into '{ptr_type}'"
                    )
                return
            var_type = env.lookup(stmt.name)
            if not var_type:
                raise TypeError(f"Assign to undeclared variable '{stmt.name}'")
            global_decl = next((g for g in prog.globals if g.name == stmt.name), None)
            if global_decl and global_decl.nomd:
                raise TypeError(f"Cannot assign to 'nomd' global variable '{stmt.name}'")
            expr_type = check_expr(stmt.expr)
            if expr_type != var_type:
                raise TypeError(f"Assign type mismatch: {var_type} = {expr_type}")
            if stmt.name in crumb_map:
                rmax, wmax, rc, wc = crumb_map[stmt.name]
                crumb_map[stmt.name] = (rmax, wmax, rc, wc + 1)
        elif isinstance(stmt, DerefStmt):
            ptr_typ = env.lookup(stmt.varname)
            if ptr_typ is None:
                raise TypeError(f"Cannot deref undeclared variable '{stmt.varname}'")
            if not ptr_typ.endswith('*'):
                raise TypeError(f"Dereferencing non-pointer variable '{stmt.varname}' of type '{ptr_typ}'")
            env.declare(stmt.varname, "undefined")
        elif isinstance(stmt, CrumbleStmt):
            if env.lookup(stmt.name) is None:
                raise TypeError(f"Cannot crumble undeclared variable '{stmt.name}'")
            if stmt.name in crumb_map:
                raise TypeError(f"Variable '{stmt.name}' already crumbled")
            crumb_map[stmt.name] = (stmt.max_reads, stmt.max_writes, 0, 0)
        elif isinstance(stmt, IndexAssign):
            arr_name = stmt.array
            var_type = env.lookup(arr_name)
            if not var_type:
                raise TypeError(f"Index‐assign to undeclared variable '{arr_name}'")
            if '[' not in var_type or not var_type.endswith(']'):
                raise TypeError(f"Index‐assign to non-array variable '{var_type}'")
            base_type = var_type.split('[', 1)[0]
            idx_type = check_expr(stmt.index)
            if idx_type != 'int':
                raise TypeError(f"Array index must be 'int', got '{idx_type}'")
            val_type = check_expr(stmt.value)
            if val_type != base_type:
                raise TypeError(f"Index‐assign type mismatch: array of {base_type}, got {val_type}")
        elif isinstance(stmt, IfStmt):
            cond_type = check_expr(stmt.cond)
            if cond_type != 'bool':
                raise TypeError(f"If condition must be bool, got {cond_type}")
            env.push()
            for s in stmt.then_body:
                check_stmt(s, expected_ret)
            env.pop()
            if stmt.else_body:
                env.push()
                if isinstance(stmt.else_body, list):
                    for s in stmt.else_body:
                        check_stmt(s, expected_ret)
                else:
                    check_stmt(stmt.else_body, expected_ret)
                env.pop()
        elif isinstance(stmt, WhileStmt):
            cond_type = check_expr(stmt.cond)
            if cond_type != 'bool':
                raise TypeError(f"While condition must be bool, got {cond_type}")
            env.push()
            for s in stmt.body:
                check_stmt(s, expected_ret)
            env.pop()
        elif isinstance(stmt, ReturnStmt):
            if stmt.expr:
                actual = check_expr(stmt.expr)
                common = unify_int_types(actual, expected_ret)
                if actual != expected_ret and (not common or common != expected_ret):
                    raise TypeError(f"Return type mismatch: expected {expected_ret}, got {actual}")
            else:
                if expected_ret != 'void':
                    raise TypeError(f"Return without value in function returning {expected_ret}")
        elif isinstance(stmt, ExprStmt):
            check_expr(stmt.expr)
        elif isinstance(stmt, Match):
            enum_typ = check_expr(stmt.expr)
            if enum_typ not in enum_defs:
                raise TypeError(f"Cannot match on non-enum type '{enum_typ}'")
            enum_def = enum_defs[enum_typ]
            defined_variants = {v.name for v in enum_def.variants}
            seen_variants = set()
            for case in stmt.cases:
                if case.variant not in defined_variants:
                    raise TypeError(f"Enum '{enum_typ}' has no variant '{case.variant}'")
                variant_info = next(v for v in enum_def.variants if v.name == case.variant)
                payload_type = variant_info.typ
                if payload_type is None and case.binding is not None:
                    raise TypeError(f"Variant '{case.variant}' carries no payload; remove binding")
                if payload_type is not None and case.binding is None:
                    raise TypeError(f"Variant '{case.variant}' requires binding of type '{payload_type}'")
                env.push()
                if case.binding is not None:
                    env.declare(case.binding, payload_type)
                for s in case.body:
                    check_stmt(s, expected_ret)
                env.pop()
                seen_variants.add(case.variant)
            if seen_variants != defined_variants:
                missing = defined_variants - seen_variants
                raise TypeError(f"Non‐exhaustive match on '{enum_typ}', missing {missing}")
        else:
            raise TypeError(f"Unsupported statement: {stmt}")
    for func in prog.funcs:
        env.push()
        for (param_typ, param_name) in func.params:
            env.declare(param_name, param_typ)
        for s in (func.body or []):
            check_stmt(s, func.ret_type)
        env.pop()
        for name, (rmax, wmax, rc, wc) in crumb_map.items():
            if rmax is not None and rc > rmax:
                raise TypeError(f"[Crawl-Checker]-[ERR]: Too many reads of '{name}': {rc} > {rmax}")
            if wmax is not None and wc > wmax:
                raise TypeError(f"[Crawl-Checker]-[ERR]: Too many writes of '{name}': {wc} > {wmax}")
            if rmax is not None and rc < rmax:
                print(f"[Crawl-Checker]-[WARN]: unused read crumbs on '{name}': {rmax - rc} left. [This is not an error but a security warning!]")
            if wmax is not None and wc < wmax:
                print(f"[Crawl-Checker]-[WARN]: unused write crumbs on '{name}': {wmax - wc} left. [This is not an error but a security warning!]")
        crumb_map.clear()
def load_extensions(config_path="ORCC.config"):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    for fname in config.get("extensions", {}).get("load", []):
        parse_modcat_file(fname)
def parse_modcat_file(fname):
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    block = None
    reg_id = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("llvm.globals"):
            block = "llvm.globals"
            continue
        elif line.startswith("reg("):
            reg_id = int(line[4:line.index(")")])
            extension_registry["registrations"][reg_id] = {}
            block = f"reg:{reg_id}"
            continue
        elif line.startswith(f"reg({reg_id}).actions"):
            block = f"reg:{reg_id}.actions"
            extension_registry["registrations"][reg_id]["actions"] = []
            continue
        elif line == "}":
            block = None
            continue
        if block == "llvm.globals":
            extension_registry["llvm.globals"].append(line)
        elif block == f"reg:{reg_id}":
            if "syntax" not in extension_registry["registrations"][reg_id]:
                extension_registry["registrations"][reg_id]["syntax"] = []
            if line.startswith("kw:"):
                kw_val = line.split(":", 1)[1].strip().strip('"')
                extension_registry["registrations"][reg_id]["kw"] = kw_val
                KEYWORDS.add(kw_val)
            elif line.startswith("type:"):
                extension_registry["registrations"][reg_id]["type"] = line.split(":", 1)[1].strip()
            elif line.startswith("syntax"):
                pass
            else:
                extension_registry["registrations"][reg_id]["syntax"].append(line)
        elif block == f"reg:{reg_id}.actions":
            extension_registry["registrations"][reg_id]["actions"].append(line)
def main():
    parser = argparse.ArgumentParser(description="Orcat Compiler")
    parser.add_argument("input", help="Input source file (.orcat or .sorcat)")
    parser.add_argument("-o", "--output", required=True, help="Output LLVM IR file (.ll)")
    parser.add_argument("--config", help="Path to extension config file", default=None)
    args = parser.parse_args()
    global compiled
    compiled = args.input
    if args.config:
        try:
            load_extensions(args.config)
        except Exception as e:
            print(f"[ORCatCompiler-WARN]: Failed to load config '{args.config}': {e}")
    else:
        print(f"[ORCatCompiler-INFO]: Skipping config load. '{args.config}' not found and not explicitly passed.")
    with open(args.input, encoding="utf-8", errors="ignore") as f:
        src = f.read()
    tokens = lex(src)
    parser_obj = Parser(tokens)
    main_prog = parser_obj.parse()
    seen_imports = set()
    seen_func_signatures = set()
    all_funcs = []
    all_structs = []
    all_enums = []
    all_globals = []
    def load_imports_recursively(prog):
        nonlocal all_funcs, all_structs, all_enums, all_globals
        for imp in prog.imports:
            candidates = []
            if imp.endswith(".or"):
                candidates.append(imp + "cat")
            elif imp.endswith(".sor"):
                candidates.append(imp + "cat")
            elif imp.endswith(".orcat") or imp.endswith(".sorcat"):
                candidates.append(imp)
            else:
                candidates += [imp + ".orcat", imp + ".sorcat"]
            resolved_path = None
            for path in candidates:
                if os.path.exists(path):
                    resolved_path = os.path.abspath(path)
                    break
            if not resolved_path:
                raise RuntimeError(f"Import '{imp}' not found. Tried: {candidates}")
            if resolved_path in seen_imports:
                continue
            seen_imports.add(resolved_path)
            with open(resolved_path, 'r', encoding="utf-8", errors="ignore") as f:
                imported_src = f.read()
            imported_tokens = lex(imported_src)
            imported_parser = Parser(imported_tokens)
            sub_prog = imported_parser.parse()
            load_imports_recursively(sub_prog)
            all_structs.extend(sub_prog.structs)
            all_enums.extend(sub_prog.enums)
            all_globals.extend(sub_prog.globals)
            for func in sub_prog.funcs:
                if func.access == 'pub':
                    sig = (func.name, len(func.params), func.is_extern)
                    if sig not in seen_func_signatures:
                        all_funcs.append(func)
                        seen_func_signatures.add(sig)
    load_imports_recursively(main_prog)
    all_funcs.extend(main_prog.funcs)
    all_structs.extend(main_prog.structs)
    all_enums.extend(main_prog.enums)
    all_globals.extend(main_prog.globals)
    final_prog = Program(
        funcs=all_funcs,
        structs=all_structs,
        enums=all_enums,
        globals=all_globals,
        imports=main_prog.imports
    )
    check_types(final_prog)
    llvm = compile_program(final_prog)
    with open(args.output, 'w', encoding="utf-8", errors="ignore") as f:
        f.write(llvm)
if __name__ == "__main__":
    main()
