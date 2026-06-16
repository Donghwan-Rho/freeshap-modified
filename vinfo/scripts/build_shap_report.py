# -*- coding: utf-8 -*-
"""Bundle the 3 Shapley-value comparison figures + explanations into one PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, PageBreak, HRFlowable)
import xml.sax.saxutils as su

pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
pdfmetrics.registerFont(UnicodeCIDFont('HYGothic-Medium'))
KO='HYSMyeongJo-Medium'; KOG='HYGothic-Medium'
INK=colors.HexColor('#1a1a1a'); ACC=colors.HexColor('#0b5394'); GREEN=colors.HexColor('#1a9850')
RED=colors.HexColor('#b2182b'); GREY=colors.HexColor('#555555'); EXBG=colors.HexColor('#fdeef0')

st={
 'title':ParagraphStyle('t',fontName=KOG,fontSize=18,leading=24,textColor=ACC,spaceAfter=8),
 'h':ParagraphStyle('h',fontName=KOG,fontSize=14,leading=19,textColor=ACC,spaceBefore=4,spaceAfter=6),
 'body':ParagraphStyle('b',fontName=KO,fontSize=10,leading=15.5,textColor=INK,spaceAfter=5,alignment=4),
 'small':ParagraphStyle('s',fontName=KO,fontSize=8.7,leading=12.5,textColor=GREY,spaceAfter=4),
 'ex':ParagraphStyle('e',fontName=KO,fontSize=9.6,leading=15,textColor=INK,backColor=EXBG,
                     borderColor=colors.HexColor('#e3b9c0'),borderWidth=0.6,borderPadding=(6,7,6,7),
                     spaceBefore=3,spaceAfter=6),
 'cap':ParagraphStyle('c',fontName=KO,fontSize=8.3,leading=11,textColor=GREY,spaceBefore=3,spaceAfter=2,alignment=1),
}
def esc(s): return su.escape(str(s))
def g(t): return f'<font name="{KOG}">{esc(t)}</font>'
def green(t): return f'<font name="{KOG}" color="#1a9850">{esc(t)}</font>'
def red(t): return f'<font name="{KOG}" color="#b2182b">{esc(t)}</font>'

story=[]
def P(t,s='body'): story.append(Paragraph(t,st[s]))
def SP(h=5): story.append(Spacer(1,h))
def HR(): story.append(HRFlowable(width='100%',thickness=0.6,color=colors.HexColor('#cccccc'),spaceBefore=4,spaceAfter=6))
def FIG(path,cap,maxw=165*mm,maxh=158*mm):
    iw,ih=ImageReader(path).getSize(); r=iw/ih
    w=maxw; h=w/r
    if h>maxh: h=maxh; w=h*r
    story.append(Image(path,width=w,height=h))
    story.append(Paragraph(cap,st['cap']))

V='/home/donghwan/freeshap/vinfo'

# ===================== PAGE 1: overview =====================
P('Shapley Value 비교 리포트: inv vs eigen / nystrom','title')
P('원본(FreeShap, ' + g('inv') + ' = full eNTK)으로 구한 점별 Shapley 값을 ' + g('기준') +
  '으로, 근사 방법인 ' + g('eigen') + '과 ' + g('nystrom') + '이 ' + g('같은 데이터 포인트') +
  '에서 값을 얼마나 다르게 추정하는지 비교한다. (랭킹이 아니라 값 자체의 차이)')
SP()
P('실험 설정','h')
P('· 데이터셋 7개: sst2, mnli, ag_news, mr, qqp, rte, mrpc &nbsp;&nbsp;· approx rank: 1%, 5%, 20%', 'small')
P('· eigen λ=1e-2 / nystrom λ = 1e-2 … 1e-6 (5종) / 공통 seed=2024, tmc=500', 'small')
P('· 점별 Shapley 값 = dv_result[:,1,:].sum(axis=1) (검증셋 accuracy 기여의 합)', 'small')
P('· 정렬: 같은 seed라 모든 방법의 sampled_idx가 동일 → 같은 위치 = 같은 데이터 포인트 (로드 시 일치 확인)', 'small')
SP(8)
P('핵심 결론','h')
P('① ' + green('eigen은 inv 값에 충실하다') + ' — 상관 0.56~0.93, 점별 평균오차 0.06~0.18, 폭발 없음.')
P('② ' + red('nystrom은 값이 폭발한다') + ' — 특히 rank 5%·작은 λ에서 Shapley 값이 ±10~30까지 튀고(정상 ~±1), '
  'inv와의 상관이 거의 0(≈0.03), 표준편차가 inv의 8~40배.')
P('③ 원인은 앞서 분석한 ' + g('m≈d 수치 불안정성') + ' — rank 5%(=차원 d≈250)에서 TMC coalition이 '
  'm≈d를 지날 때 feature 역행렬이 폭발 → marginal 기여 폭발. ' + g('λ가 작을수록(정규화 약할수록) 심함') + '.')
P('④ 주의: 이건 ' + g('"값"의 불안정') + '이지 "랭킹"이 아니다. 데이터 선택(랭킹 기반)은 멀쩡해 보여도, '
  'Shapley 값을 직접 쓰는 분석에는 nystrom(특히 mid-rank·작은 λ)이 부적합하다.', 'small')
story.append(PageBreak())

# ===================== PAGE A: data-selection accuracy (ranking) =====================
P('A. 데이터 선택 정확도 (랭킹 기반)','h')
P('각 방법의 Shapley 랭킹으로 상위 k% 데이터를 고른 뒤 ' + g('full eNTK(inv)') + '로 정확도를 측정한 것. '
  'x=선택 비율(1~100%), y=검증 정확도. eigen λ=1e-2 + nystrom 5λ. 곡선이 위일수록 그 랭킹이 더 좋은 데이터를 골랐다는 뜻.')
P('핵심 — 대부분의 칸에서 곡선들이 비슷하게 겹친다 = ' + g('랭킹 기반 선택 정확도는 방법 간 큰 차이가 없다') +
  '. 뒤(①②③)에서 값이 폭발했던 칸(mr·qqp rank 5%)에서도 이 정확도 곡선은 멀쩡하다 → ' +
  green('값은 못 믿어도 랭킹(=상대 순서)은 대체로 살아남는다') + '. 그래서 값 비교를 따로 봐야 한다.')
P('예시 — ' + g('ag_news rank 20%') + '에서는 eigen(검정)이 nystrom보다 대체로 위 = 고rank에선 eigen 우세. '
  '방법 간 차이는 주로 ' + g('저선택 구간(그래프 좌측)') + '에서 드러난다.', 'ex')
FIG(f'{V}/eigen_vs_nystrom_acc_comparison.png','선택 비율별 정확도 (full eNTK 평가) — eigen λ=1e-2 + nystrom 5λ',
    maxw=150*mm, maxh=165*mm)
story.append(PageBreak())

# ===================== PAGE 2: heatmap =====================
P('① 충실도 히트맵 (correlation & mean|Δ|)','h')
P('한 칸 = (데이터셋, rank) × 방법. ' + g('왼쪽') + ' = inv와의 상관(1=거의 동일, 0=무관), '
  + g('오른쪽') + ' = 점별 평균 절대오차 mean|approx−inv|. ' + green('초록일수록 inv와 가깝다') +
  '(왼쪽은 1에 가까움, 오른쪽은 0에 가까움). 빨강은 멀다.')
P('맨 왼쪽 열(eigen)이 전반적으로 초록인 반면, 오른쪽으로(λ가 작아질수록) · 특히 ' + g('rank 5% 행') +
  '에서 새빨갛게 변한다 = nystrom 값 폭발.')
P('예시 — ' + g('ag_news rank 5%') + ' 행을 보면: eigen corr=' + green('0.90') +
  '(진초록)인데, 같은 행 nys λ=1e-6은 corr=' + red('0.03') + '(새빨강). '
  '왼→오로 갈수록 빨개진다. 반대로 ' + g('rte') + ' 행들은 nystrom도 비교적 초록 = 이 데이터셋은 폭발이 없다.', 'ex')
FIG(f'{V}/shap_value_fidelity_heatmap.png','correlation(좌) · mean|Δ|(우) 히트맵 — 초록=inv와 가까움',
    maxw=170*mm, maxh=150*mm)
story.append(PageBreak())

# ===================== PAGE 3: absdiff bars =====================
P('② 평균·최대 오차 막대 (log 스케일)','h')
P('각 칸 = (데이터셋, rank)의 6개 방법. ' + g('막대 높이 = 평균오차 mean|Δ|') + ', ' +
  g('막대 위 가로 tick = 최대오차 max|Δ|') + '. y축은 log 스케일이고, ' + red('빨간 점선') +
  ' = 정상 Shapley 값 스케일(≈1). 막대가 빨간선을 넘으면 사실상 폭발이다.')
P('eigen(검정) 막대는 어디서나 빨간선 한참 아래 = 평균으로도 최댓값으로도 inv에 가깝다. '
  'nystrom 막대는 ' + g('rank 5%(가운데 열)') + '에서 빨간선 위로 치솟는다.')
P('예시 — ' + g('mr rank 5%') + ': eigen은 평균 ' + green('0.09') + ' · 최대 ' + green('0.72') +
  '(빨간선 아래)인데, nys λ=1e-2는 평균 ' + red('2.59') + ' · 최대 ' + red('12.96') +
  '(빨간선 위). 즉 ' + g('평균은 ~29배, 최대는 ~18배') + ' eigen이 더 작다 = inv에 더 비슷하다. '
  '다만 ' + g('rte rank 5%') + '는 nystrom도 빨간선 아래(안정)라, 폭발이 데이터셋·rank에 따라 갈린다.', 'ex')
FIG(f'{V}/shap_value_absdiff_summary.png','막대=mean|Δ|, tick=max|Δ| (log y; 빨간선≈정상 sv 스케일)',
    maxw=150*mm, maxh=170*mm)
story.append(PageBreak())

# ===================== PAGE 4: error dist =====================
P('③ 점별 오차 분포','h')
P('Shapley 값 자체가 아니라 ' + g('점별 오차 = (approx 값 − inv 값)') + '의 분포(히스토그램)다. '
  'x=오차(0이면 inv와 일치), 검정=eigen. ' + g('|오차|>1 인 점은 ±1로 클리핑') +
  '해 양끝에 쌓이게 했다(각 칸 좌상단에 클리핑된 비율 표시).')
P(green('0에 좁고 뾰족할수록 inv에 충실') + '하다. eigen은 어디서나 0 중심의 뾰족한 종 모양. '
  'nystrom은 안정 칸에선 조금 더 퍼진 종, ' + red('폭발 칸에선 0 근처가 비고 양끝(±1)에 스파이크') + '가 생긴다.')
P('예시 — ' + g('qqp rank 5%') + ': eigen은 0 중심의 뾰족한 종이지만, nystrom은 대부분의 점이 ±1로 '
  '클리핑돼 양끝에 몰린다(값이 inv와 완전히 어긋남). 반대로 ' + g('sst2 rank 1%') +
  '는 eigen·nystrom 모두 0 근처 종 모양 = 둘 다 안정적.', 'ex')
FIG(f'{V}/shap_value_error_dist.png','(approx − inv) 점별 오차 분포 — x는 ±1로 클리핑, 폭발분은 양끝에 쌓임',
    maxw=150*mm, maxh=168*mm)
story.append(PageBreak())

# ===================== PAGE 4: ranking heatmap =====================
P('④ 순위(ranking) 보존도 — Spearman & top-5% 겹침','h')
P('①②③은 ' + g('값') + '의 유사도였다. 이 페이지는 ' + g('순위(순서)가 inv와 같은지') + '를 직접 본다. ' +
  g('왼쪽') + ' = Spearman 순위상관(1=같은 순서, ~0=무관), ' + g('오른쪽') +
  ' = inv가 고른 상위 5% 데이터를 얼마나 똑같이 고르나(top-5% 집합 겹침; 무작위면 ~0.05). ' +
  green('초록일수록 순위를 잘 보존') + '.')
P('핵심 — Spearman이 ①의 Pearson을 거의 그대로 따라간다. 즉 ' + g('값이 터진 칸은 순위도 같이 무너진다') +
  '(순위상관 ≈0, top-5% 겹침이 무작위 수준 ~0.05). eigen은 값뿐 아니라 ' + green('순위도 잘 보존') + '한다.')
P('예시 — ' + g('mr rank 5%') + ': eigen은 Spearman ' + green('0.76') + ' · top-5% 겹침 ' + green('0.59') +
  '인데, nys λ=1e-2는 Spearman ' + red('0.04') + ' · top-5% 겹침 ' + red('0.06') +
  '(= 거의 무작위). 즉 폭발 칸의 nystrom 순위는 inv와 ' + g('사실상 무관') + '하다.', 'ex')
FIG(f'{V}/shap_value_ranking_heatmap.png','Spearman 순위상관(좌) · top-5% 겹침(우) — 초록=순위를 잘 보존',
    maxw=170*mm, maxh=150*mm)
story.append(PageBreak())

# ===================== PAGE 5: takeaway =====================
P('정리 & 시사점','h')
P('· ' + green('eigen') + '은 Shapley ' + g('값과 순위 둘 다') + ' inv에 충실 → ' +
  g('값 기반·순위 기반 분석 모두 안전') + '.')
P('· ' + red('nystrom') + '은 mid-rank·작은 λ에서 ' + g('값이 폭발하고 순위(④)도 inv와 무관') +
  '해진다. 데이터 선택 정확도(A)가 그래도 괜찮아 보인 건 ' +
  g('무작위에 가까운 선택도 정확도가 그럭저럭 나오기 때문') + '이지 순위가 맞아서가 아니다.')
P('· nystrom을 꼭 쓴다면: ' + g('큰 λ(≥1e-2)') + '를 쓰고, rank를 m≈d 충돌 구간(여기선 5%≈d250 부근)을 '
  '피하도록 선택. 그래도 mr·qqp처럼 λ=1e-2에서도 터지는 경우가 있으니, ' + g('값 정밀 비교에는 eigen 권장') + '.')
P('· 같은 경향이 데이터셋 전반(7개)에서 일관되게 나타난다. 예외적으로 ' + g('rte(가장 작은 데이터셋, d가 작음)') +
  '는 nystrom도 비교적 안정적이다.', 'small')
SP(8)
P('※ 그림은 모두 seed=2024, tmc=500 기준. 표/수치 원본은 분석 스크립트로 언제든 재생성 가능.', 'small')

def footer(c,d):
    c.saveState(); c.setFont(KO,8); c.setFillColor(GREY)
    c.drawString(20*mm,12*mm,'Shapley value 비교: inv vs eigen / nystrom')
    c.drawRightString(190*mm,12*mm,f'p.{d.page}'); c.restoreState()

doc=BaseDocTemplate(f'{V}/shap_value_comparison_report.pdf',pagesize=A4,
                    leftMargin=20*mm,rightMargin=20*mm,topMargin=16*mm,bottomMargin=18*mm,
                    title='Shapley value comparison report')
fr=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='n')
doc.addPageTemplates([PageTemplate(id='m',frames=[fr],onPage=footer)])
doc.build(story)
print('saved:',f'{V}/shap_value_comparison_report.pdf')
