import sys
from typing import List, Optional, Tuple, Union, Dict
from dataclasses import dataclass
import copy
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
    'fn', 'fnmacro', 'if', 'else', 'while', 'return',
    'import', 'pub', 'priv', 'prot', 'extern',
    'int', 'int8', 'int16', 'int32', 'int64',
    'float', 'bool', 'char', 'string', 'void',
    'true', 'false', 'struct', 'enum', 'match', 'nomd',
    'pin'
}
SINGLE_CHARS = {
    '(': 'LPAREN',   ')': 'RPAREN',   '{': 'LBRACE',   '}': 'RBRACE',
    ',': 'COMMA',    ';': 'SEMI',
    '=': 'EQUAL',    '+': 'PLUS',     '-': 'MINUS',    '*': 'STAR',
    '/': 'SLASH',    '<': 'LT',       '>': 'GT',       '[': 'LBRACKET',
    ']': 'RBRACKET', '?': 'QUESTION', '.': 'DOT', ':': 'COLON', '%': 'PERCENT'
}

MULTI_CHARS = {
    '==': 'EQEQ', '!=': 'NEQ', '<=': 'LE', '>=': 'GE', '->': 'ARROW', '&&': 'AND',  '||': 'OR'
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
class GlobalVar:
    typ: str
    name: str
    expr: Optional[Expr]
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
class VarDecl(Stmt):
    access: str
    typ: str
    name: str
    expr: Optional[Expr]
    nomd: bool = False
@dataclass
class Assign(Stmt): name: str; expr: Expr
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
class MacroDef(Stmt):
    name: str
    params: List[str]
    body: List[Stmt]
@dataclass
class Index(Expr):
    array: Expr
    index: Expr
@dataclass
class MacroUse(Stmt):
    name: str
    args: List[Expr]
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
    macros: List[MacroDef]
    structs: List[StructDef]
    enums: List[EnumDef]
    globals: List[GlobalVar]
string_constants: List[str] = []
struct_field_map: Dict[str, List[str]] = {}
generated_mono: Dict[str, bool] = {}
all_funcs: List[Func] = []
enum_variant_map: Dict[str, List[Tuple[str, Optional[str]]]] = {}
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
        macros = []
        structs = []
        enums = []
        globals = []
        while self.peek().kind != 'EOF':
            if self.match('IMPORT'):
                path = self.expect('STRING').value
                self.expect('SEMI')
                imports.append(path)
            elif self.match('PIN'):
                globals.append(self.parse_global())
            elif self.peek().kind == 'NOMD':
                decl = self.parse_var_decl()
                globals.append(GlobalVar(decl.typ, decl.name, decl.expr))
            elif self.peek().kind == 'FNMACRO':
                macros.append(self.parse_macro_def())
            elif self.peek().kind == 'STRUCT':
                structs.append(self.parse_struct_def())
            elif self.peek().kind == 'ENUM':
                enums.append(self.parse_enum_def())
            else:
                funcs.append(self.parse_func())
        return Program(funcs, imports, macros, structs, enums, globals)
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
        access = 'priv'
        is_extern = False
        while True:
            tk = self.peek().kind
            if tk == 'EXTERN':
                is_extern = True
                self.bump()
                continue
            if tk in {'PUB', 'PRIV', 'PROT'}:
                access = self.bump().kind.lower()
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
                    pname = self.expect('IDENT').value
                    params.append((typ, pname))
                else:
                    raise SyntaxError(
                        f"Expected type, got {self.peek().kind} "
                        f"at {self.peek().line}:{self.peek().col}"
                    )
                if not self.match('COMMA'):
                    break
        self.expect('RPAREN')
        self.expect('LT')
        ret_type = self.bump().value
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
        if t.kind == 'IDENT' and self.tokens[self.pos + 1].kind == 'EQUAL':
            return self.parse_assign()
        if t.kind == 'IDENT' and self.tokens[self.pos + 1].kind == 'LBRACKET':
            return self.parse_index_assign()
        if t.kind == 'IDENT' and self.tokens[self.pos + 1].kind == 'LPAREN':
            return self.parse_expr_stmt()
        if t.kind in {'PUB', 'PRIV', 'PROT', 'NOMD'} or t.kind in TYPE_TOKENS or t.kind == 'IDENT':
            return self.parse_var_decl()
        if t.kind == 'IF':
            return self.parse_if()
        if t.kind == 'WHILE':
            return self.parse_while()
        if t.kind == 'RETURN':
            return self.parse_return()
        return self.parse_expr_stmt()
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
            raise RuntimeError(f"Cannot assign to 'nomd' variable '{name}'")
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
            'STAR': 5,
            'SLASH': 5,
            'PERCENT': 5,
            'PLUS': 4,
            'MINUS': 4,
            'LT': 3,
            'LE': 3,
            'GT': 3,
            'GE': 3,
            'EQEQ': 2,
            'NEQ': 2,
            'AND': 1,
            'OR': 0
        }.get(op, 0)
    def parse_primary(self) -> Expr:
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
            if t.kind == 'LPAREN':
                expr = self.parse_expr()
                self.expect('RPAREN')
                return expr
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
def unify_int_types(t1: str, t2: str) -> Optional[str]:
    rank = {
        "int8": 1, "int16": 2, "int32": 3, "int64": 4,
        "int": 4
    }
    for t in [t1, t2]:
        if t not in rank and not t.startswith("int"):
            return None
    return max(t1, t2, key=lambda t: rank.get(t, 0))
def unify_types(t1: str, t2: str) -> Optional[str]:
    if t1 == t2:
        return t1
    int_common = unify_int_types(t1, t2)
    if int_common:
        return int_common
    if (t1, t2) in {("float", "int"), ("int", "float")}:
        return "float"
    return None
type_map = {
    'int': 'i64',
    'int8': 'i8',
    'int16': 'i16',
    'int32': 'i32',
    'int64': 'i64',
    'float': 'double',
    'bool': 'i1',
    'char': 'i8',
    'string': 'i8*',
    'void': 'void'
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
        base_ty_start = llvm_ty.find('x') + 2
        base_ty = llvm_ty[base_ty_start:-1]
        tmp_ptr = new_tmp()
        tmp_val = new_tmp()
        out.append(f"  {tmp_ptr} = getelementptr inbounds {llvm_ty}, {llvm_ty}* %{name}_addr, i32 0, i32 {idx}")
        out.append(f"  {tmp_val} = load {base_ty}, {base_ty}* {tmp_ptr}")
        return tmp_val
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
            f"[{length} x i8]* {label}, i32 0, i32 0")
        return tmp
    if isinstance(expr, Var):
        result = symbol_table.lookup(expr.name)
        if result is None:
            raise RuntimeError(f"Undefined variable: {expr.name}")
        typ, name = result
        tmp = new_tmp()
        if name.startswith('@'):
            out.append(f"  {tmp} = load {typ}, {typ}* {name}")
        else:
            if name.startswith('@'):
                out.append(f"  {tmp} = load {typ}, {typ}* {name}")
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
                '+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv',
                '==': 'fcmp oeq', '!=': 'fcmp one',
                '<': 'fcmp olt', '<=': 'fcmp ole', '>': 'fcmp ogt', '>=': 'fcmp oge'
            }.get(expr.op)
        else:
            op = {
                '+': 'add', '-': 'sub', '*': 'mul', '/': 'sdiv',
                '==': 'icmp eq', '!=': 'icmp ne',
                '<': 'icmp slt', '<=': 'icmp sle', '>': 'icmp sgt', '>=': 'icmp sge'
            }.get(expr.op)
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
        return llvm_ty
    if isinstance(expr, FieldAccess):
        return infer_type(expr)
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
def gen_stmt(stmt: Stmt, out: List[str]):
    if isinstance(stmt, VarDecl):
        if isinstance(stmt, VarDecl):
            if "[" in stmt.typ:
                base, count = stmt.typ.split("[")
                count = count[:-1]
                llvm_base = type_map[base]
                llvm_ty = f"[{count} x {llvm_base}]"
            elif stmt.typ in type_map:
                llvm_ty = type_map[stmt.typ]
            else:
                llvm_ty = f"%struct.{stmt.typ}"
            out.append(f"  %{stmt.name}_addr = alloca {llvm_ty}")
            symbol_table.declare(stmt.name, llvm_ty, stmt.name)
            if stmt.expr:
                val = gen_expr(stmt.expr, out)
                src_type = type_map[infer_type(stmt.expr)]
                dst_type = llvm_ty
                if src_type != dst_type:
                    def bitsize(ty: str) -> int:
                        return int(ty[1:])
                    src_bits = bitsize(src_type)
                    dst_bits = bitsize(dst_type)
                    cast_tmp = new_tmp()
                    if src_bits > dst_bits:
                        out.append(f"  {cast_tmp} = trunc {src_type} {val} to {dst_type}")
                    else:
                        out.append(f"  {cast_tmp} = sext {src_type} {val} to {dst_type}")
                    val = cast_tmp
                out.append(f"  store {dst_type} {val}, {dst_type}* %{stmt.name}_addr")
    elif isinstance(stmt, Assign):
        val = gen_expr(stmt.expr, out)
        llvm_ty, ir_name = symbol_table.lookup(stmt.name)
        if ir_name.startswith('@'):
            out.append(f"  store {llvm_ty} {val}, {llvm_ty}* {ir_name}")
        else:
            out.append(f"  store {llvm_ty} {val}, {llvm_ty}* %{ir_name}_addr")
    elif isinstance(stmt, IndexAssign):
        idx = gen_expr(stmt.index, out)
        val = gen_expr(stmt.value, out)
        llvm_ty, name = symbol_table.lookup(stmt.array)
        base_ty_start = llvm_ty.find('x') + 2
        base_ty = llvm_ty[base_ty_start:-1]
        tmp_ptr = new_tmp()
        out.append(f"  {tmp_ptr} = getelementptr inbounds {llvm_ty}, {llvm_ty}* %{name}_addr, i32 0, i32 {idx}")
        out.append(f"  store {base_ty} {val}, {base_ty}* {tmp_ptr}")
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
                gen_stmt(s, out)
            out.append(f"  br label %{end_lbl}")
        out.append(f"{end_lbl}:")
def gen_func(fn: Func) -> List[str]:
    if fn.type_params:
        return []
    if fn.is_extern:
        param_sig = ", ".join(f"{type_map[t]}" for t, _ in fn.params)
        ret_ty = type_map[fn.ret_type]
        generated_mono[fn.name] = True
        return [f"declare {ret_ty} @{fn.name}({param_sig})"]
    symbol_table.push()
    ret_ty = type_map[fn.ret_type]
    generated_mono[fn.name] = True
    param_sig = ", ".join(f"{type_map[t]} %{n}" for t, n in fn.params)
    out: List[str] = [f"define {ret_ty} @{fn.name}({param_sig}) {{", "entry:"]
    for typ, name in fn.params:
        llvm_ty = type_map[typ]
        out.append(f"  %{name}_addr = alloca {llvm_ty}")
        out.append(f"  store {llvm_ty} %{name}, {llvm_ty}* %{name}_addr")
        symbol_table.declare(name, llvm_ty, name)
    for stmt in fn.body:
        gen_stmt(stmt, out)
    llvm_ret = type_map[fn.ret_type]
    if llvm_ret == 'void':
        out.append("  ret void")
    elif llvm_ret == 'double':
        out.append(f"  ret {llvm_ret} 0.0")
    else:
        if llvm_ret.startswith('i'):
            out.append(f"  ret {llvm_ret} 0")
        else:
            out.append(f"  ret {llvm_ret} null")
    out.append("}")
    symbol_table.pop()
    return out
def compile_program(prog: Program) -> str:
    global all_funcs, func_table
    all_funcs = prog.funcs[:]
    string_constants.clear()
    func_table.clear()
    for fn in prog.funcs:
        llvm_ret_ty = type_map[fn.ret_type]
        func_table[fn.name] = llvm_ret_ty
    lines: List[str] = [
        "; ModuleID = 'orcat'",
        "source_filename = \"main.orcat\"",
        ""]
    for g in prog.globals:
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
        struct_field_map[sdef.name] = [fld.name for fld in sdef.fields]
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
    for fn in prog.funcs:
        lines += gen_func(fn)
    if string_constants:
        lines.extend(string_constants)
        lines.append("")
        string_constants.clear()
    return "\n".join(lines)
def check_types(prog: Program):
    env = TypeEnv()
    funcs = {f.name: f for f in prog.funcs}
    for g in prog.globals:
        env.declare(g.name, g.typ)
    struct_defs: Dict[str, StructDef] = {s.name: s for s in prog.structs}
    enum_defs:   Dict[str, EnumDef]   = {e.name: e for e in prog.enums}
    struct_field_map: Dict[str, List[Tuple[str, str]]] = {}
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
        if isinstance(expr, Var):
            typ = env.lookup(expr.name)
            if not typ:
                raise TypeError(f"Use of undeclared variable '{expr.name}'")
            return typ
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
            if expr.name not in funcs:
                raise TypeError(f"Call to undeclared function '{expr.name}'")
            fn = funcs[expr.name]
            if len(expr.args) != len(fn.params):
                raise TypeError(f"Arity mismatch in call to '{expr.name}'")
            for arg_expr, (expected_type, _) in zip(expr.args, fn.params):
                actual_type = check_expr(arg_expr)
                common = unify_int_types(actual_type, expected_type)
                if actual_type != expected_type and (not common or common != expected_type):
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
            return expr.name
        raise TypeError(f"Unsupported expression: {expr}")
    def check_stmt(stmt: Stmt, expected_ret: str):
        if isinstance(stmt, VarDecl):
            if env.lookup(stmt.name):
                raise TypeError(f"Variable '{stmt.name}' already declared")
            raw_typ = stmt.typ
            if '[' in raw_typ and raw_typ.endswith(']'):
                base_type = raw_typ.split('[', 1)[0]
            else:
                base_type = raw_typ
            if base_type not in type_map and base_type not in struct_defs and base_type not in enum_defs:
                raise TypeError(f"Unknown type '{raw_typ}'")
            env.declare(stmt.name, raw_typ)
            if stmt.expr:
                expr_type = check_expr(stmt.expr)
                if isinstance(stmt.expr, IntLit):
                    int_targets = {"int8", "int16", "int32", "int64"}
                    if raw_typ in int_targets:
                        return
                common = unify_int_types(expr_type, raw_typ)
                if expr_type != raw_typ and (not common or common != raw_typ):
                    raise TypeError(
                        f"Type mismatch in variable init '{stmt.name}': expected {raw_typ}, got {expr_type}")
        elif isinstance(stmt, Assign):
            var_type = env.lookup(stmt.name)
            if not var_type:
                raise TypeError(f"Assign to undeclared variable '{stmt.name}'")
            expr_type = check_expr(stmt.expr)
            if expr_type != var_type:
                raise TypeError(f"Assign type mismatch: {var_type} = {expr_type}")
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
                        else_list.extend(substitute_stmt(s, mapping) for _ in [s])
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
                            f"got {len(call.args)}")
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
                Func(fn.access, fn.name, fn.type_params, fn.params, fn.ret_type, expanded, fn.is_extern))
        else:
            new_funcs.append(fn)
    return Program(new_funcs, prog.imports, [], prog.structs, prog.enums, prog.globals)
def main():
    if len(sys.argv) != 4 or sys.argv[2] != "-o":
        print("Usage: ORCC.exe {filename}.orcat|.sorcat -o {filename}.ll")
        return
    inp = sys.argv[1]
    outp = sys.argv[3]
    with open(inp, encoding="utf-8", errors="ignore") as f:
        src = f.read()
    tokens = lex(src)
    parser = Parser(tokens)
    prog = parser.parse()
    prog = expand_macros(prog)
    all_funcs = []
    all_macros = []
    seen_imports = set()
    for imp in prog.imports:
        if imp in seen_imports:
            continue
        seen_imports.add(imp)
        with open(imp, 'r', encoding="utf-8", errors="ignore") as f:
            imported_src = f.read()
        imported_tokens = lex(imported_src)
        imported_parser = Parser(imported_tokens)
        sub_prog = imported_parser.parse()
        sub_prog = expand_macros(sub_prog)
        check_types(sub_prog)
        for func in sub_prog.funcs:
            if func.access == 'pub':
                all_funcs.append(func)
        for macro in sub_prog.macros:
            all_macros.append(macro)
    prog.funcs = all_funcs + prog.funcs
    prog.macros = all_macros + prog.macros
    check_types(prog)
    llvm = compile_program(prog)
    with open(outp, 'w', encoding="utf-8", errors="ignore") as f:
        f.write(llvm)
if __name__ == "__main__":
    main()
