from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "paper_draft" / "CIRCA_RT_RTSS_DRAFT_V1_20260526.md"
OUT_DIR = ROOT / "paper_draft" / "latex_rtss"
TEX_PATH = OUT_DIR / "circa_rt_rtss_draft.tex"
MAIN_TEX_PATH = OUT_DIR / "circa_rt_rtss_main_submission.tex"


TITLE = "CIRCA-RT: Deadline-Safe Selective Auditing for Real-Time Perception Pipelines"


MATH_TOKENS = [
    ("C_A^I", r"$C_A^{I}$"),
    ("C_M^I", r"$C_M^{I}$"),
    ("C_P^I", r"$C_P^{I}$"),
    ("R_i^I", r"$R_i^{I}$"),
    ("S_i^D", r"$S_i^{D}$"),
    ("S_i^N", r"$S_i^{N}$"),
    ("A_i", r"$A_i$"),
    ("M_i", r"$M_i$"),
    ("P_i", r"$P_i$"),
    ("H_i", r"$H_i$"),
    ("B_i", r"$B_i$"),
    ("D_i", r"$D_i$"),
    ("N_i", r"$N_i$"),
    ("r_i", r"$r_i$"),
    ("d_i", r"$d_i$"),
    ("t_i", r"$t_i$"),
    ("q_i", r"$q_i$"),
    ("u_i", r"$u_i$"),
    ("b_i^+", r"$b_i^{+}$"),
    ("b_i", r"$b_i$"),
    ("b_{i+1}", r"$b_{i+1}$"),
    ("q_A", r"$q_A$"),
    ("theta_l", r"$\theta_l$"),
    ("theta_h", r"$\theta_h$"),
    ("rho", r"$\rho$"),
    ("beta", r"$\beta$"),
    ("Pi_s", r"$\Pi_s$"),
    ("N_A", r"$N_A$"),
    ("W_A^I", r"$W_A^{I}$"),
    ("DBF_A^I", r"$DBF_A^{I}$"),
    ("Delta", r"$\Delta$"),
    ("B_max", r"$B_{\max}$"),
    ("U_A^avg", r"$U_A^{avg}$"),
    ("v_i", r"$v_i$"),
    ("alpha", r"$\alpha$"),
]


def protect_inline_code(text: str, placeholders: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = f"@@CODE{len(placeholders)}@@"
        value = match.group(1).replace("\\", r"\textbackslash{}")
        placeholders[key] = r"\texttt{" + latex_escape(value) + "}"
        return key

    return re.sub(r"`([^`]+)`", repl, text)


def protect_math_tokens(text: str, placeholders: dict[str, str]) -> str:
    # Longer tokens first, otherwise S_i may steal part of S_i^D.
    for raw, tex in sorted(MATH_TOKENS, key=lambda x: len(x[0]), reverse=True):
        key = f"@@MATH{len(placeholders)}@@"
        pattern = re.escape(raw)
        text, n = re.subn(pattern, key, text)
        if n:
            placeholders[key] = tex
    return text


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def tex_text(text: str) -> str:
    placeholders: dict[str, str] = {}
    text = protect_inline_code(text, placeholders)
    text = protect_math_tokens(text, placeholders)
    text = latex_escape(text)
    text = restore_placeholders(text, placeholders)
    text = text.replace("--", "--")
    return text


def tex_heading(line: str) -> str | None:
    if line.startswith("## "):
        title = line[3:].strip()
        if title == "Abstract":
            return None
        if title == "Appendix":
            return r"\appendices" + "\n" + r"\section{Appendix}"
        title = re.sub(r"^[IVX]+\.\s+", "", title)
        return r"\section{" + tex_text(title) + "}"
    if line.startswith("### "):
        title = line[4:].strip()
        title = re.sub(r"^[A-Z]\.\s+", "", title)
        return r"\subsection{" + tex_text(title) + "}"
    return None


def mathify(expr: str) -> str:
    s = expr.strip().rstrip(",")
    replacements = [
        ("theta_l", r"\theta_l"),
        ("theta_h", r"\theta_h"),
        ("Delta", r"\Delta"),
        ("beta", r"\beta"),
        ("rho", r"\rho"),
        ("Pi_s", r"\Pi_s"),
        ("C_A^I", r"C_A^{I}"),
        ("C_M^I", r"C_M^{I}"),
        ("C_P^I", r"C_P^{I}"),
        ("R_i^I", r"R_i^{I}"),
        ("S_i^D", r"S_i^{D}"),
        ("S_i^N", r"S_i^{N}"),
        ("W_A^I", r"W_A^{I}"),
        ("DBF_A^I", r"DBF_A^{I}"),
        ("N_A", r"N_A"),
        ("B_max", r"B_{\max}"),
        ("U_A^avg", r"U_A^{avg}"),
        ("v_i", r"v_i"),
        ("alpha", r"\alpha"),
        ("q_A", r"q_A"),
        ("d_i", r"d_i"),
        ("t_i", r"t_i"),
        ("q_i", r"q_i"),
        ("b_i", r"b_i"),
        ("u_i", r"u_i"),
        ("H_i", r"H_i"),
        ("B_i", r"B_i"),
        ("D_i", r"D_i"),
        ("N_i", r"N_i"),
    ]
    for raw, tex in replacements:
        s = s.replace(raw, tex)
    s = s.replace("<=", r"\le")
    s = s.replace(">=", r"\ge")
    s = s.replace("*", r"\cdot")
    s = s.replace(" := ", r" \triangleq ")
    s = s.replace(":=", r"\triangleq")
    s = s.replace(" = ", " = ")
    return s


def equation_block(block: str) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    body = r" \\".join(mathify(ln) for ln in lines)
    return "\\begin{equation}\n\\begin{aligned}\n" + body + "\n\\end{aligned}\n\\end{equation}"


def notation_table(block: str) -> str:
    rows = []
    for ln in block.splitlines():
        if not ln.strip() or ln.startswith("Notation") or ln.startswith("Symbol"):
            continue
        parts = re.split(r"\s{2,}", ln.strip(), maxsplit=1)
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    out = [
        r"\begin{table}[t]",
        r"\caption{Notation used in the problem formulation.}",
        r"\label{tab:notation}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabularx}{\columnwidth}{@{}lX@{}}",
        r"\toprule",
        r"Symbol & Description \\",
        r"\midrule",
    ]
    for sym, desc in rows:
        out.append(tex_text(sym) + " & " + tex_text(desc) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(out)


def table1() -> str:
    rows = [
        ("AGX real-frame main", "AV2, DROID, nuImages", "MobileNetV2, ResNet18, ResNet50", "Direct AGX audit admission"),
        ("Backend matrix", "Real-frame replay", "4 models; CUDA/ONNX/TRT", "Backend sensitivity"),
        ("Reduced pressure", "AGX pressure traces", "72 real-frame rows; 672 bound rows", "Pressure validation"),
        ("ROS2/DDS boundary", "Publisher/subscriber", "MobileNetV2; Python JPEG/String DDS", "Integration boundary"),
        ("Server replay", "Existing AGX scores", "DS, SS, CBS variants", "Scheduling-side baseline"),
    ]
    out = [
        r"\begin{table*}[t]",
        r"\caption{Experimental matrix and claim boundary.}",
        r"\label{tab:exp-matrix}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{@{}lXXX@{}}",
        r"\toprule",
        r"Path & Source & Configuration & Claim supported \\",
        r"\midrule",
    ]
    for row in rows:
        out.append(" & ".join(tex_text(x) for x in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(out)


def table2() -> str:
    rows = [
        ("AlwaysAudit", "1.0000", "1.0000", "18.190 ms", "0.00031", "0.00281"),
        ("ContextAwareConformal", "0.4969", "0.2224", "17.940 ms", "0.00031", "0.00281"),
        ("ConditionalRFF-HSIC", "0.4533", "0.2853", "17.227 ms", "0.00024", "0.00274"),
        ("CIRCA-RT", "0.5384", "0.0483", "16.076 ms", "0.00005", "0.00052"),
        ("CIRCA-RT-Slack", "0.5384", "0.0483", "16.071 ms", "0.00000", "0.00000"),
    ]
    out = [
        r"\begin{table}[t]",
        r"\caption{Selected main AGX real-frame aggregate.}",
        r"\label{tab:main-agx}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Recall & Audit & p99 & Mean miss & Max miss \\",
        r"\midrule",
    ]
    for row in rows:
        out.append(" & ".join(tex_text(x) for x in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table3() -> str:
    rows = [
        ("CIRCA-RT-Slack", "0.0444", "0.0781", "0.000000", "0.0000", "0.0000"),
        ("Risk+DS", "0.0446", "0.0805", "0.000176", "0.1470", "0.1470"),
        ("Risk+DS+Slack", "0.0444", "0.0800", "0.000000", "0.0000", "0.0000"),
        ("Risk+SS", "0.0397", "0.0671", "0.000176", "0.1326", "0.1326"),
        ("Risk+SS+Slack", "0.0395", "0.0666", "0.000000", "0.0000", "0.0000"),
        ("Risk+CBS", "0.0086", "0.0117", "0.000032", "0.0179", "0.0179"),
        ("Risk+CBS+Slack", "0.0086", "0.0116", "0.000000", "0.0000", "0.0000"),
    ]
    out = [
        r"\begin{table*}[t]",
        r"\caption{Offline scheduling-side replay over existing AGX traces.}",
        r"\label{tab:server-replay}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Audit rate & Audit hit & Deadline miss & Bound fail & Carry-in \\",
        r"\midrule",
    ]
    for row in rows:
        out.append(" & ".join(tex_text(x) for x in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out)


def algorithm1() -> str:
    return r"""
\begin{algorithm}[t]
\caption{CIRCA-RT online audit admission.}
\label{alg:circa-admit}
\scriptsize
\begin{algorithmic}[1]
\REQUIRE frame $i$; risk score $q_i$; thresholds $\theta_l,\theta_h$; bucket level $b_i$ and parameters $\beta,\rho,q_A$; measured required-path cost $P_i$; monitor bound $C_M^I$; audit bound $C_A^I$; deadline $D$; guard $G$; optional next-release slack $S_i^N$
\ENSURE lightweight alarm $a_i$; audit decision $u_i$; updated bucket $b_{i+1}$
\STATE \textbf{function} \textsc{AdmitAudit}$(i,q_i,b_i)$
\STATE $b_i \leftarrow \min\{\beta,b_i+\rho\}$ \hfill $\triangleright$ budget replenishment
\STATE $a_i \leftarrow \mathbf{1}[q_i \ge \theta_l]$ \hfill $\triangleright$ lightweight alarm
\STATE $u_i \leftarrow 0$
\STATE $S_i^D \leftarrow D-P_i-C_M^I$ \hfill $\triangleright$ Eq.~(1), direct form
\STATE $H_i \leftarrow (q_i \ge \theta_h)$ \hfill $\triangleright$ Eq.~(2a)
\STATE $B_i \leftarrow (b_i \ge q_A)$ \hfill $\triangleright$ Eq.~(2b)
\STATE $D_i \leftarrow (C_A^I+G \le S_i^D)$ \hfill $\triangleright$ Eq.~(2c)
\STATE $N_i \leftarrow \mathrm{true}$
\IF{next-release isolation is enabled}
  \STATE $N_i \leftarrow (C_A^I+G \le S_i^N)$ \hfill $\triangleright$ Eq.~(3)
\ENDIF
\IF{$H_i \wedge B_i \wedge D_i \wedge N_i$}
  \STATE $u_i \leftarrow 1$
  \STATE $b_i \leftarrow b_i-q_A$ \hfill $\triangleright$ consume audit charge
\ENDIF
\STATE $b_{i+1} \leftarrow b_i$
\STATE \textbf{return} $(a_i,u_i,b_{i+1})$
\STATE \textbf{end function}
\end{algorithmic}
\end{algorithm}
""".strip()


def algorithm2() -> str:
    return r"""
\begin{algorithm}[t]
\caption{Offline replay of budgeted optional-work servers.}
\label{alg:server-replay}
\scriptsize
\begin{algorithmic}[1]
\REQUIRE scored trace $\mathcal{T}$ with $q_i,P_i,C_M^I,C_A^I,D,G$; server type $s\in\{\mathrm{DS},\mathrm{SS},\mathrm{CBS}\}$; server budget $Q$; replenishment period $\Pi_s$; audit charge $q_A$; threshold $\theta_h$; slack flag $z$
\ENSURE replayed audit decisions $u_i$ and aggregate timing metrics
\STATE \textbf{function} \textsc{ServerReplay}$(\mathcal{T},s,z)$
\STATE initialize server state for $s$
\FOR{each frame $i$ in $\mathcal{T}$ in release order}
  \STATE $\textsc{UpdateServer}_s(i,Q,\Pi_s)$ \hfill $\triangleright$ DS/SS/CBS rule
  \STATE $H_i \leftarrow (q_i \ge \theta_h)$ \hfill $\triangleright$ Eq.~(2a)
  \STATE $B_i \leftarrow \textsc{CanPay}_s(q_A)$ \hfill $\triangleright$ server budget test
  \STATE $D_i \leftarrow \mathrm{true}$
  \IF{$z=\mathrm{true}$}
    \STATE $S_i^D \leftarrow D-P_i-C_M^I$ \hfill $\triangleright$ Eq.~(1), replay form
    \STATE $D_i \leftarrow (C_A^I+G \le S_i^D)$ \hfill $\triangleright$ Eq.~(2c)
  \ENDIF
  \STATE $u_i \leftarrow 0$
  \IF{$H_i \wedge B_i \wedge D_i$}
    \STATE $u_i \leftarrow 1$
    \STATE $\textsc{ChargeServer}_s(q_A)$
  \ENDIF
  \STATE $L_i \leftarrow P_i+C_M^I+u_iC_A^I$
  \STATE record $L_i$, deadline miss, bound failure, and carry-in
\ENDFOR
\STATE \textbf{return} audit rate, audit-hit rate, deadline misses, bound failures, carry-in violations
\STATE \textbf{end function}
\end{algorithmic}
\end{algorithm}
""".strip()


def figure_block(alt: str, path: str) -> str:
    caption = alt
    caption = re.sub(r"^Figure\s+\d+\.\s*", "", caption)
    fig_no = re.search(r"Figure\s+(\d+)", alt)
    label = "fig:" + (fig_no.group(1) if fig_no else re.sub(r"\W+", "-", caption.lower()).strip("-"))
    p = Path(path)
    pdf_path = p.with_suffix(".pdf")
    use_path = pdf_path if (ROOT / "paper_draft" / pdf_path).exists() else p
    use_path_posix = use_path.as_posix()
    is_wide = (
        "figures/structure/" in use_path_posix
        or "figures/str/" in use_path_posix
        or "multipanel" in use_path_posix
        or "baseline" in use_path_posix
    )
    width = r"\textwidth" if is_wide else r"\columnwidth"
    env = "figure*" if width == r"\textwidth" else "figure"
    latex_path = Path("..") / use_path
    return "\n".join([
        rf"\begin{{{env}}}[t]",
        r"\centering",
        rf"\includegraphics[width={width}]{{{latex_path.as_posix()}}}",
        r"\caption{" + tex_text(caption) + "}",
        rf"\label{{{label}}}",
        rf"\end{{{env}}}",
    ])


def code_block_to_tex(block: str) -> str:
    stripped = block.strip()
    if stripped.startswith("Notation summary"):
        return notation_table(stripped)
    if stripped.startswith("Algorithm 1"):
        return algorithm1()
    if stripped.startswith("Algorithm 2"):
        return algorithm2()
    if stripped.startswith("Table 1"):
        return table1()
    if stripped.startswith("Table 2"):
        return table2()
    if stripped.startswith("Table 3"):
        return table3()
    return equation_block(stripped)


def paragraph_to_tex(paragraph: str) -> str:
    p = paragraph.strip()
    if not p:
        return ""
    m = re.match(r"^\*\*(.+?)\*\*\s*(.*)$", p)
    if m:
        return r"\noindent\textbf{" + tex_text(m.group(1)) + "} " + tex_text(m.group(2))
    return tex_text(p)


def build_tex(include_appendix: bool = True) -> str:
    md = MD_PATH.read_text(encoding="utf-8")
    lines = md.splitlines()
    abstract_lines: list[str] = []
    body_lines: list[str] = []
    in_abstract = False
    skip_title = True
    for line in lines:
        if not include_appendix and line.strip() == "## Appendix":
            break
        if skip_title and line.startswith("# "):
            skip_title = False
            continue
        if line.strip() == "## Abstract":
            in_abstract = True
            continue
        if in_abstract and line.startswith("## "):
            in_abstract = False
        if in_abstract:
            abstract_lines.append(line)
        else:
            body_lines.append(line)

    out: list[str] = [
        r"\documentclass[10pt,conference]{IEEEtran}",
        r"\IEEEoverridecommandlockouts",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{cite}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{algorithm}",
        r"\usepackage{algorithmic}",
        r"\usepackage{graphicx}",
        r"\usepackage{textcomp}",
        r"\usepackage{xcolor}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{array}",
        r"\usepackage{balance}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\newcommand{\todo}[1]{\textcolor{red}{#1}}",
        r"\begin{document}",
        r"\title{" + TITLE + r"}",
        r"\author{\IEEEauthorblockN{Anonymous Authors}\\\IEEEauthorblockA{Paper under double-anonymous review}}",
        r"\maketitle",
        r"\begin{abstract}",
        tex_text(" ".join(x.strip() for x in abstract_lines if x.strip())),
        r"\end{abstract}",
        r"\begin{IEEEkeywords}",
        r"real-time systems, runtime monitoring, admission control, edge perception, ROS2, GPU scheduling",
        r"\end{IEEEkeywords}",
    ]

    in_code = False
    in_refs = False
    code_lines: list[str] = []
    para: list[str] = []

    def flush_para() -> None:
        nonlocal para
        if para:
            out.append(paragraph_to_tex(" ".join(para)))
            para = []

    for line in body_lines:
        if in_refs:
            if line.startswith("## ") and line.strip() != "## References":
                out.append(r"\end{thebibliography}")
                in_refs = False
                heading = tex_heading(line)
                if heading is not None:
                    out.append(heading)
                    continue
            else:
                ref = re.match(r"^\[(\d+)\]\s+(.+)$", line.strip())
                if ref:
                    out.append(r"\bibitem{ref" + ref.group(1) + "} " + tex_text(ref.group(2)))
                elif line.strip():
                    out.append(tex_text(line.strip()))
                continue
        if line.startswith("```"):
            if in_code:
                out.append(code_block_to_tex("\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                flush_para()
                in_code = True
                code_lines = []
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.strip() == "## References":
            flush_para()
            out.append(r"\begin{thebibliography}{99}")
            in_refs = True
            continue
        heading = tex_heading(line)
        if heading is not None:
            flush_para()
            out.append(heading)
            continue
        img = re.match(r"!\[(.+?)\]\((.+?)\)", line.strip())
        if img:
            flush_para()
            out.append(figure_block(img.group(1), img.group(2)))
            continue
        if not line.strip():
            flush_para()
            continue
        # Keep contribution list compact.
        if re.match(r"^\d+\.\s+", line.strip()):
            flush_para()
            out.append(r"\noindent " + tex_text(line.strip()))
            continue
        para.append(line.strip())

    flush_para()
    if in_refs:
        out.append(r"\end{thebibliography}")
    out += [
        r"\balance",
        r"\end{document}",
    ]
    return "\n\n".join(x for x in out if x is not None and x != "")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex = build_tex(include_appendix=True)
    TEX_PATH.write_text(tex, encoding="utf-8")
    main_tex = build_tex(include_appendix=False)
    MAIN_TEX_PATH.write_text(main_tex, encoding="utf-8")
    print(TEX_PATH)
    print(MAIN_TEX_PATH)


if __name__ == "__main__":
    main()
