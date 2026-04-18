import re, pathlib

src = pathlib.Path('/Users/ari/hadoop-crime-project/softwarex_paper.tex').read_text()

# 1. Replace the tikzpicture block (and its surrounding \begin{center}...\end{center}) with an includegraphics figure.
tikz_pattern = re.compile(
    r'\\begin\{center\}\s*\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}\s*\\end\{center\}',
    re.DOTALL,
)
replacement = (
    r'\\begin{figure}[H]' "\n"
    r'\\centering' "\n"
    r'\\includegraphics[width=\\textwidth]{architecture_diagram.png}' "\n"
    r'\\caption{Three-stage PySpark pipeline architecture with HDFS storage and shared utilities.}' "\n"
    r'\\label{fig:architecture}' "\n"
    r'\\end{figure}'
)
new_src, n1 = tikz_pattern.subn(replacement, src)
assert n1 == 1, f"tikz replacement count = {n1}"

# 2. Rewrite the comparison table using simple Yes / -- markers (drop pifont/\cmark)
new_table = r'''\begin{table}[h!]
\small
\centering
\renewcommand{\arraystretch}{1.25}
\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}X|c|c|c|c|c|}
\hline
\textbf{Feature} & \textbf{Ours} & \textbf{Pandas} & \textbf{Tableau} & \textbf{QGIS} & \textbf{R/tidyverse} \\
\hline
NCRB-specific harmonization & Yes & -- & -- & -- & -- \\
\hline
Distributed (Spark/HDFS) & Yes & -- & -- & -- & -- \\
\hline
ML clustering + forecasting & Yes & Partial & -- & -- & Yes \\
\hline
Interactive choropleth maps & Yes & -- & Yes & Yes & Partial \\
\hline
Open source & Yes & Yes & -- & Yes & Yes \\
\hline
Docker one-command deploy & Yes & -- & -- & -- & -- \\
\hline
End-to-end pipeline & Yes & -- & -- & -- & -- \\
\hline
\end{tabularx}
\caption{Feature comparison with existing tools for Indian crime data analysis.}
\end{table}'''

old_table_pattern = re.compile(
    r'\\begin\{table\}\[h!\]\s*\\small\s*\\centering\s*\\begin\{tabularx\}\{\\textwidth\}\{\|l\|c\|c\|c\|c\|c\|\}.*?Feature comparison with existing tools.*?\\end\{table\}',
    re.DOTALL,
)
new_src, n2 = old_table_pattern.subn(lambda _m: new_table, new_src)
assert n2 == 1, f"table replacement count = {n2}"

out = pathlib.Path('/Users/ari/hadoop-crime-project/softwarex_paper_docx.tex')
out.write_text(new_src)
print(f"ok: tikz={n1} table={n2} -> {out}")
