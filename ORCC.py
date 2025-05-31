import sys
from typing import List, Optional, Tuple, Union, Dict
from dataclasses import dataclass
import copy
TYPE_TOKENS = {'IDENT', 'INT', 'FLOAT', 'STRING', 'BOOL', 'CHAR', 'VOID'}
string_constants: List[str] = []
@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int
KEYWORDS = {
    'fn', 'fnmacro', 'if', 'else', 'while', 'return',
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
        if c == '/' and i + 1 < len(source) and source[i+1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
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
class BinOp(Expr): op: str; left: Expr; right: Expr
@dataclass
class Call(Expr): name: str; args: List[Expr]
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
class MacroDef(Stmt):
    name: str
    params: List[str]
    body: List[Stmt]
@dataclass
class MacroUse(Stmt):
    name: str
    args: List[Expr]
@dataclass
class Func:
    access: str
    name: str
    params: List[Tuple[str, str]]
    ret_type: str
    body: Optional[List[Stmt]] = None
    is_extern: bool = False
@dataclass
class Program:
    funcs: List[Func]
    imports: List[str]
    macros: List[MacroDef]
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
        macros = []
        while self.peek().kind != 'EOF':
            if self.match('IMPORT'):
                path = self.expect('STRING').value
                self.expect('SEMI')
                imports.append(path)
            elif self.peek().kind == 'FNMACRO':
                macros.append(self.parse_macro_def())
            else:
                funcs.append(self.parse_func())
        return Program(funcs, imports, macros)
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
                if self.peek().kind in TYPE_TOKENS:
                    typ = self.bump().value
                else:
                    raise SyntaxError(f"Expected type, got {self.peek().kind} at "
                                      f"{self.peek().line}:{self.peek().col}")
                self.expect('COLON')
                pname = self.expect('IDENT').value
                params.append((typ, pname))
                if not self.match('COMMA'):
                    break
        self.expect('RPAREN')
        self.expect('LT')
        if self.peek().kind in TYPE_TOKENS:
            ret_type = self.bump().value
        else:
            raise SyntaxError(f"Expected return type, got {self.peek().kind} at "
                              f"{self.peek().line}:{self.peek().col}")
        self.expect('GT')
        if is_extern:
            self.expect('SEMI')
            return Func(access, name, params, ret_type, None, True)
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
        if t.kind in {'PUB', 'PRIV', 'PROT'} or (t.kind in TYPE_TOKENS and t.kind != 'IDENT'):
            return self.parse_var_decl()
        if t.kind == 'IDENT' and self.tokens[self.pos + 1].kind == 'EQUAL':
            return self.parse_assign()
        if t.kind == 'IDENT' and self.tokens[self.pos + 1].kind == 'LPAREN':
            return self.parse_expr_stmt()
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
        if self.peek().kind in TYPE_TOKENS and self.peek().kind != 'IDENT':
            typ = self.bump().value
        else:
            raise SyntaxError(
                f"Expected type (one of {TYPE_TOKENS}), got {self.peek().kind} "
                f"at {self.peek().line}:{self.peek().col}"
            )
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
    def parse_expr(self, min_prec: int = 0) -> Expr:
        left = self.parse_primary()
        while True:
            op_token = self.peek()
            if op_token.kind not in {'PLUS', 'MINUS', 'STAR', 'SLASH', 'EQEQ', 'NEQ', 'LT', 'LE', 'GT', 'GE'}:
                break
            op_prec = self.get_precedence(op_token.kind)
            if op_prec < min_prec:
                break
            self.bump()
            right = self.parse_expr(op_prec + 1)
            left = BinOp(op_token.value, left, right)
        return left
    def get_precedence(self, op: str) -> int:
        return {
            'STAR': 5, 'SLASH': 5,
            'PLUS': 4, 'MINUS': 4,
            'EQEQ': 3, 'NEQ': 3,
            'LT': 2, 'LE': 2, 'GT': 2, 'GE': 2,
        }.get(op, 0)
    def parse_primary(self) -> Expr:
        t = self.bump()
        if t.kind == 'INT':
            return IntLit(int(t.value))
        if t.kind == 'FLOAT':
            return FloatLit(float(t.value))
        if t.kind == 'STRING':
            return StrLit(t.value)
        if t.kind == 'CHAR':
            return CharLit(t.value)
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
    def parse_macro_def(self) -> MacroDef:
        self.expect('FNMACRO')
        name = self.expect('IDENT').value
        self.expect('LPAREN')
        params: List[str] = []
        if self.peek().kind != 'RPAREN':
            while True:
                params.append(self.expect('IDENT').value)
                if not self.match('COMMA'):
                    break
        self.expect('RPAREN')
        self.expect('LBRACE')
        body_stmts: List[Stmt] = []
        while self.peek().kind != 'RBRACE':
            body_stmts.append(self.parse_stmt())
        self.expect('RBRACE')
        return MacroDef(name, params, body_stmts)
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
symbol_table = SymbolTable()
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
        raw = expr.value
        esc = ""
        for ch in raw:
            if ch == '\n':
                esc += r'\0A'
            elif ch == '\t':
                esc += r'\09'
            elif ch == '\\':
                esc += r'\\'
            elif ch == '"':
                esc += r'\"'
            else:
                esc += ch
        length = len(raw) + 1
        string_constants.append(
            f'{label} = private unnamed_addr constant [{length} x i8] c"{esc}\\00"'
        )
        out.append(
            f"  {tmp} = getelementptr inbounds [{length} x i8], "
            f"[{length} x i8]* {label}, i32 0, i32 0"
        )
        return tmp
    if isinstance(expr, Var):
        result = symbol_table.lookup(expr.name)
        if result is None:
            raise RuntimeError(f"Undefined variable: {expr.name}")
        typ, name = result
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
            return ''
        else:
            tmp = new_tmp()
            out.append(f"  {tmp} = call {ret_ty} @{expr.name}({', '.join(args_ir)})")
            return tmp
    raise RuntimeError(f"Unhandled expr: {expr}")
def infer_type(expr: Expr) -> str:
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
    if isinstance(expr, Var):
        result = symbol_table.lookup(expr.name)
        if result is None:
            raise RuntimeError(f"Undefined variable: {expr.name}")
        llvm_ty, _ = result
        for k, v in type_map.items():
            if v == llvm_ty:
                return k
        return llvm_ty
    if isinstance(expr, BinOp):
        left_type = infer_type(expr.left)
        right_type = infer_type(expr.right)
        if left_type != right_type:
            raise RuntimeError(f"Type mismatch in binary op '{expr.op}': {left_type} vs {right_type}")
        return left_type
    if isinstance(expr, Call):
        if expr.name not in func_table:
            raise RuntimeError(f"Call to undefined function '{expr.name}'")
        ret_llvm_ty = func_table[expr.name]
        for k, v in type_map.items():
            if v == ret_llvm_ty:
                return k
        return ret_llvm_ty
    raise RuntimeError(f"Cannot infer type for expression: {expr}")
def gen_stmt(stmt: Stmt, out: List[str]):
    if isinstance(stmt, VarDecl):
        llvm_ty = type_map[stmt.typ]
        out.append(f"  %{stmt.name}_addr = alloca {llvm_ty}")
        symbol_table.declare(stmt.name, llvm_ty, stmt.name)
        if stmt.expr:
            val = gen_expr(stmt.expr, out)
            out.append(f"  store {llvm_ty} {val}, {llvm_ty}* %{stmt.name}_addr")
    elif isinstance(stmt, Assign):
        val = gen_expr(stmt.expr, out)
        llvm_ty, _ = symbol_table.lookup(stmt.name)
        out.append(f"  store {llvm_ty} {val}, {llvm_ty}* %{stmt.name}_addr")
    elif isinstance(stmt, IfStmt):
        cond = gen_expr(stmt.cond, out)
        then_lbl = new_label('then')
        else_lbl = new_label('else') if stmt.else_body else None
        end_lbl = new_label('endif')
        out.append(f"  br i1 {cond}, label %{then_lbl}, label %{else_lbl or end_lbl}")
        out.append(f"{then_lbl}:")
        symbol_table.push()
        for s in stmt.then_body:
            gen_stmt(s, out)
        symbol_table.pop()
        out.append(f"  br label %{end_lbl}")
        if stmt.else_body:
            out.append(f"{else_lbl}:")
            symbol_table.push()
            if isinstance(stmt.else_body, list):
                for s in stmt.else_body:
                    gen_stmt(s, out)
            elif isinstance(stmt.else_body, IfStmt):
                gen_stmt(stmt.else_body, out)
            symbol_table.pop()
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
        symbol_table.push()
        for s in stmt.body:
            gen_stmt(s, out)
        symbol_table.pop()
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
    if fn.name in func_table:
        return []
    if fn.is_extern:
        param_sig = ", ".join(f"{type_map[t]}" for t, _ in fn.params)
        ret_ty = type_map[fn.ret_type]
        func_table[fn.name] = ret_ty
        return [f"declare {ret_ty} @{fn.name}({param_sig})"]
    symbol_table.push()
    ret_ty = type_map[fn.ret_type]
    func_table[fn.name] = ret_ty
    param_sig = ", ".join(f"{type_map[t]} %{n}" for t, n in fn.params)
    out: List[str] = [f"define {ret_ty} @{fn.name}({param_sig}) {{", "entry:"]
    for typ, name in fn.params:
        llvm_ty = type_map[typ]
        out.append(f"  %{name}_addr = alloca {llvm_ty}")
        out.append(f"  store {llvm_ty} %{name}, {llvm_ty}* %{name}_addr")
        symbol_table.declare(name, llvm_ty, name)
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
    symbol_table.pop()
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
        sub_prog = expand_macros(sub_prog)
        check_types(sub_prog)
        for f in sub_prog.funcs:
            lines += gen_func(f)
    for fn in prog.funcs:
        lines += gen_func(fn)
    return "\n".join([
        "; ModuleID = 'orcat'",
        'source_filename = "main.orcat"',
        "",
        *string_constants,
        *lines
    ])
def check_types(prog: Program):
    env = TypeEnv()
    funcs = {f.name: f for f in prog.funcs}
    def check_expr(expr: Expr) -> str:
        if isinstance(expr, IntLit): return 'int'
        if isinstance(expr, FloatLit): return 'float'
        if isinstance(expr, BoolLit): return 'bool'
        if isinstance(expr, CharLit): return 'char'
        if isinstance(expr, StrLit): return 'string'
        if isinstance(expr, Var):
            typ = env.lookup(expr.name)
            if not typ:
                raise TypeError(f"Use of undeclared variable '{expr.name}'")
            return typ
        if isinstance(expr, BinOp):
            left = check_expr(expr.left)
            right = check_expr(expr.right)
            if left != right:
                raise TypeError(f"Type mismatch: {left} {expr.op} {right}")
            return left
        if isinstance(expr, Call):
            if expr.name not in funcs:
                raise TypeError(f"Call to undeclared function '{expr.name}'")
            fn = funcs[expr.name]
            if len(expr.args) != len(fn.params):
                raise TypeError(f"Arity mismatch in call to '{expr.name}'")
            for arg, (expected_type, _) in zip(expr.args, fn.params):
                actual_type = check_expr(arg)
                if actual_type != expected_type:
                    raise TypeError(
                        f"Argument type mismatch in call to '{expr.name}': expected {expected_type}, got {actual_type}")
            return fn.ret_type
        raise TypeError(f"Unsupported expression: {expr}")
    def check_stmt(stmt: Stmt, expected_ret: str):
        if isinstance(stmt, VarDecl):
            if env.lookup(stmt.name):
                raise TypeError(f"Variable '{stmt.name}' already declared")
            if stmt.typ not in type_map:
                raise TypeError(f"Unknown type '{stmt.typ}'")
            env.declare(stmt.name, stmt.typ)
            if stmt.expr:
                expr_type = check_expr(stmt.expr)
                if expr_type != stmt.typ:
                    raise TypeError(
                        f"Type mismatch in variable init '{stmt.name}': expected {stmt.typ}, got {expr_type}")
        elif isinstance(stmt, Assign):
            var_type = env.lookup(stmt.name)
            if not var_type:
                raise TypeError(f"Assign to undeclared variable '{stmt.name}'")
            expr_type = check_expr(stmt.expr)
            if expr_type != var_type:
                raise TypeError(f"Assign type mismatch: {var_type} = {expr_type}")
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
                elif isinstance(stmt.else_body, IfStmt):
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
                if actual != expected_ret:
                    raise TypeError(f"Return type mismatch: expected {expected_ret}, got {actual}")
            else:
                if expected_ret != 'void':
                    raise TypeError(f"Return without value in function returning {expected_ret}")
        elif isinstance(stmt, ExprStmt):
            check_expr(stmt.expr)
    for func in prog.funcs:
        env.push()
        for typ, name in func.params:
            env.declare(name, typ)
        for stmt in (func.body or []):
            check_stmt(stmt, func.ret_type)
        env.pop()
def expand_macros(prog: Program) -> Program:
    macro_dict: Dict[str, MacroDef] = {m.name: m for m in prog.macros}

    def substitute_expr(expr: Expr, mapping: Dict[str, Expr]) -> Expr:
        if isinstance(expr, IntLit):
            return IntLit(expr.value)
        if isinstance(expr, FloatLit):
            return FloatLit(expr.value)
        if isinstance(expr, BoolLit):
            return BoolLit(expr.value)
        if isinstance(expr, CharLit):
            return CharLit(expr.value)
        if isinstance(expr, StrLit):
            return StrLit(expr.value)
        if isinstance(expr, Var):
            if expr.name in mapping:
                return copy.deepcopy(mapping[expr.name])
            return Var(expr.name)
        if isinstance(expr, BinOp):
            left_copy = substitute_expr(expr.left, mapping)
            right_copy = substitute_expr(expr.right, mapping)
            return BinOp(expr.op, left_copy, right_copy)
        if isinstance(expr, Call):
            new_args: List[Expr] = []
            for a in expr.args:
                new_args.append(substitute_expr(a, mapping))
            return Call(expr.name, new_args)
        raise RuntimeError(f"Unsupported Expr in macro substitution: {expr}")
    def substitute_stmt(stmt: Stmt, mapping: Dict[str, Expr]) -> Stmt:
        if isinstance(stmt, VarDecl):
            init_copy = substitute_expr(stmt.expr, mapping) if stmt.expr else None
            return VarDecl(stmt.access, stmt.typ, stmt.name, init_copy)
        if isinstance(stmt, Assign):
            rhs_copy = substitute_expr(stmt.expr, mapping)
            return Assign(stmt.name, rhs_copy)
        if isinstance(stmt, IfStmt):
            cond_copy = substitute_expr(stmt.cond, mapping)
            then_copy: List[Stmt] = []
            for s in stmt.then_body:
                then_copy.append(substitute_stmt(s, mapping))
            else_copy = None
            if stmt.else_body:
                if isinstance(stmt.else_body, IfStmt):
                    else_copy = substitute_stmt(stmt.else_body, mapping)
                else:
                    else_list: List[Stmt] = []
                    for s in stmt.else_body:
                        else_list.append(substitute_stmt(s, mapping))
                    else_copy = else_list
            return IfStmt(cond_copy, then_copy, else_copy)
        if isinstance(stmt, WhileStmt):
            cond_copy = substitute_expr(stmt.cond, mapping)
            body_copy: List[Stmt] = []
            for s in stmt.body:
                body_copy.append(substitute_stmt(s, mapping))
            return WhileStmt(cond_copy, body_copy)
        if isinstance(stmt, ReturnStmt):
            if stmt.expr:
                expr_copy = substitute_expr(stmt.expr, mapping)
                return ReturnStmt(expr_copy)
            return ReturnStmt(None)
        if isinstance(stmt, ExprStmt):
            new_expr = substitute_expr(stmt.expr, mapping)
            return ExprStmt(new_expr)
        raise RuntimeError(f"Unsupported Stmt in macro substitution: {stmt}")
    def expand_stmt_list(stmts: List[Stmt]) -> List[Stmt]:
        result: List[Stmt] = []
        for stmt in stmts:
            if isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Call):
                call = stmt.expr
                if call.name in macro_dict:
                    macro_def = macro_dict[call.name]
                    if len(call.args) != len(macro_def.params):
                        raise RuntimeError(
                            f"Macro '{call.name}' expects {len(macro_def.params)} args, "
                            f"got {len(call.args)}"
                        )
                    mapping: Dict[str, Expr] = {}
                    for param_name, arg_expr in zip(macro_def.params, call.args):
                        mapping[param_name] = copy.deepcopy(arg_expr)
                    for body_stmt in macro_def.body:
                        inlined = substitute_stmt(body_stmt, mapping)
                        expanded_fragment = expand_stmt_list([inlined])
                        result.extend(expanded_fragment)
                    continue
            if isinstance(stmt, IfStmt):
                new_then = expand_stmt_list(stmt.then_body)
                new_else = None
                if stmt.else_body:
                    if isinstance(stmt.else_body, IfStmt):
                        new_else = expand_stmt_list([stmt.else_body])[0]
                    else:
                        new_else_list: List[Stmt] = []
                        for s in stmt.else_body:
                            new_else_list.extend(expand_stmt_list([s]))
                        new_else = new_else_list
                result.append(IfStmt(stmt.cond, new_then, new_else))
                continue
            if isinstance(stmt, WhileStmt):
                new_body = expand_stmt_list(stmt.body)
                result.append(WhileStmt(stmt.cond, new_body))
                continue
            result.append(stmt)
        return result
    new_funcs: List[Func] = []
    for fn in prog.funcs:
        if fn.body:
            expanded = expand_stmt_list(fn.body)
            new_funcs.append(
                Func(fn.access, fn.name, fn.params, fn.ret_type, expanded, fn.is_extern)
            )
        else:
            new_funcs.append(fn)
    return Program(new_funcs, prog.imports, [])
import ctypes
def main():
    if len(sys.argv) != 4 or sys.argv[2] != "-o":
        ctypes.windll.user32.MessageBoxW(
            None,
            "Usage: ORCC.exe <input.sorcat|.orcat> -o <output.ll>",
            "ORCC.exe",
            0x10
        )
        return
    inp = sys.argv[1]
    outp = sys.argv[3]
    with open(inp) as f:
        src = f.read()
    tokens = lex(src)
    parser = Parser(tokens)
    prog = parser.parse()
    prog = expand_macros(prog)
    all_funcs = []
    all_macros = []
    for imp in prog.imports:
        with open(imp, 'r') as f:
            src = f.read()
        tokens = lex(src)
        parser = Parser(tokens)
        sub_prog = parser.parse()
        sub_prog = expand_macros(sub_prog)
        check_types(sub_prog)
        all_funcs.extend(sub_prog.funcs)
        all_macros.extend(sub_prog.macros)
    prog.funcs = all_funcs + prog.funcs
    prog.macros = all_macros + prog.macros
    check_types(prog)
    llvm = compile_program(prog)
    with open(outp, 'w') as f:
        f.write(llvm)
if __name__ == "__main__":
    main()
