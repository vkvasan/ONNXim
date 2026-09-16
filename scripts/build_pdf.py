import sys, re, pathlib
src = pathlib.Path("report.html").read_text()
title = re.search(r'<title>(.*?)</title>', src).group(1)
rest  = src[src.index('</title>')+len('</title>'):]
style_end = rest.index('</style>')+8
css_block, body_html = rest[:style_end], rest[style_end:]

OVERRIDE = """
<style>
  @font-face{font-family:"RptSans";src:url("file:///home/kevijayakumar/.claude/jobs/83d7841b/tmp/.pdfenv/lib/python3.12/site-packages/font_source_sans_pro/files/SourceSansPro-Regular.otf");font-weight:400;font-style:normal}
  @font-face{font-family:"RptSans";src:url("file:///home/kevijayakumar/.claude/jobs/83d7841b/tmp/.pdfenv/lib/python3.12/site-packages/font_source_sans_pro/files/SourceSansPro-Semibold.otf");font-weight:700;font-style:normal}
  @font-face{font-family:"RptSans";src:url("file:///home/kevijayakumar/.claude/jobs/83d7841b/tmp/.pdfenv/lib/python3.12/site-packages/font_source_sans_pro/files/SourceSansPro-It.otf");font-weight:400;font-style:italic}
  @font-face{font-family:"RptSerif";src:url("file:///home/kevijayakumar/.claude/jobs/83d7841b/tmp/.pdfenv/lib/python3.12/site-packages/font_source_serif_pro/files/SourceSerifPro-Regular.otf");font-weight:400;font-style:normal}
  @font-face{font-family:"RptSerif";src:url("file:///home/kevijayakumar/.claude/jobs/83d7841b/tmp/.pdfenv/lib/python3.12/site-packages/font_source_serif_pro/files/SourceSerifPro-Bold.otf");font-weight:700;font-style:normal}

  html,body{margin:0;padding:0}
  @page{
    size:letter; margin:0.95in 1.05in 0.85in;
    @top-left{
      content:"NPU-HBM Access-Pattern Analysis";
      font-family:"Helvetica Neue",Helvetica,Arial,sans-serif; font-size:6.8pt;
      letter-spacing:.16em; text-transform:uppercase; color:#77808D;
      margin-bottom:9pt;
    }
    @top-right{
      content:string(sectitle);
      font-family:"Helvetica Neue",Helvetica,Arial,sans-serif; font-size:6.8pt;
      letter-spacing:.16em; text-transform:uppercase; color:#77808D;
      margin-bottom:9pt;
    }
    @bottom-right{
      content:counter(page);
      font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-size:7.4pt;
      color:#77808D; margin-top:10pt;
    }
  }
  @page:first{ @top-left{content:""} @top-right{content:""} }

  h2{ string-set: sectitle content(); }

  @media print{
    body{background:#fff; padding:0; font-size:9.7pt; line-height:1.52}
    /* override the variables so every rule picks up the embedded faces */
    :root{--sans:"RptSans",sans-serif; --serif:"RptSerif",serif}
    h1{font-family:"RptSerif",serif; font-weight:400; letter-spacing:-.008em; font-size:21pt}
    .part .ptitle{font-family:"RptSerif",serif; font-weight:400; letter-spacing:-.006em}
    .sheet{border:none; margin:0; padding:0; min-height:0; max-width:none;
           display:block; break-after:auto; page-break-after:auto}
    .runhead,.folio{display:none}
    .part{break-before:page; page-break-before:always; margin-top:0}
    .sheet:first-of-type .part{break-before:auto; page-break-before:auto}
    h2,h3{break-after:avoid; page-break-after:avoid}
    p,li{orphans:3; widows:3}
    h2{break-before:auto; margin-top:22px}
    figcaption,.tabcaption{break-before:avoid}
    table,figure,.tablewrap,.note,pre{break-inside:avoid; page-break-inside:avoid}
    /* diagrams must never split */
    .sys,.seg,.segrow,.syslink,.sysrow,.memrow,.cells,.grp,.seglegend,figure
      {break-inside:avoid; page-break-inside:avoid}
    .sys{break-after:avoid; page-break-after:avoid}
    .seg{break-after:avoid; page-break-after:avoid}
    .seglegend{break-before:avoid; page-break-before:avoid}
    .titleblock{margin-bottom:20px}
    /* white paper, no tinted boxes: structure comes from rules alone */
    :root{--paper:#fff; --wash:transparent; --wash-2:transparent}
    .sheet{background:#fff !important}
    pre,th,tr.hi td,.sysrow .box,.sysbus,.grp,.grp.fill,.cells span,.cell,.note
      {background:transparent !important}
    .grp.fill .cells span{background:transparent !important}
    pre{border-left:2px solid var(--accent); padding-left:14px}
    th{border-bottom:.8px solid var(--ink)}
    .sysrow .box{border:.8px solid var(--rule)}
    .sysbus{border:1.4px solid var(--accent)}
    /* the figures' whole point is that head 1 is shaded: keep those fills */
    .cells span.on,.cell.on,.seglegend .sw.f
      {background:var(--accent) !important; color:#fff !important;
       border-color:var(--accent) !important}
    .grp.fill{border:1.6px solid var(--accent) !important}
    /* box-shadow is unsupported in print: mark highlighted rows with a rule */
    tr.hi td:first-child{border-left:2.2pt solid #2B4C7E; padding-left:8px}
    /* tables must fit the page box: overflow-x is dropped in print */
    table{font-size:7.0pt}
    th{font-size:6.4pt; padding:6px 7px 5px}
    td{padding:4px 7px}
    td.k{font-size:8.2pt}
    .tablewrap{margin:12px 0 15px}
  }
</style>
"""
doc = (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
       f'<title>{title}</title>\n{css_block}\n{OVERRIDE}\n</head><body>\n{body_html}\n</body></html>')
pathlib.Path("_print.html").write_text(doc)

from weasyprint import HTML
out = sys.argv[1]
rendered = HTML(string=doc, base_url=".").render()
print("pages:", len(rendered.pages))
rendered.write_pdf(out)
