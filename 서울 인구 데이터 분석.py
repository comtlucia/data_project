import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("📍 우리 동네 인구 구조, 데이터로 읽다")

# 파일 경로
file_gender = "202504_202504_연령별인구현황_월간_남녀구분.csv"

# CSV 불러오기
df_gender = pd.read_csv(file_gender, encoding="cp949")

# 행정구역 이름 정제: 괄호 앞 지역명만 추출
df_gender = df_gender[df_gender["행정구역"].str.contains("(\d+)", regex=True)]
df_gender["지역명"] = df_gender["행정구역"].str.split("(").str[0].str.strip()

# 동 단위만 필터링
df_gender = df_gender[df_gender["지역명"].str.endswith("동")]

# 연령 컬럼 정의
age_columns_male = [col for col in df_gender.columns if "세" in col and "_남_" in col]
age_columns_female = [col for col in df_gender.columns if "세" in col and "_여_" in col]
ages = [col.split("_")[-1] for col in age_columns_male]

# 지역 선택
selected_region = st.selectbox("📍 분석할 지역을 선택하세요:", options=df_gender["지역명"].unique())
region_data = df_gender[df_gender["지역명"] == selected_region].iloc[0]

# 값 처리 및 정수형 변환
population_male = region_data[age_columns_male].str.replace(",", "").fillna("0").astype(int).tolist()
population_female = region_data[age_columns_female].str.replace(",", "").fillna("0").astype(int).tolist()

# 총합 및 비율 계산
total_male = sum(population_male)
total_female = sum(population_female)

male_ratio = [round(p / total_male * 100, 2) for p in population_male]
female_ratio = [round(p / total_female * 100, 2) for p in population_female]

# 🎯 선택 지역 인구 피라미드
fig_pyramid = go.Figure()
fig_pyramid.add_trace(go.Bar(
    y=ages,
    x=[-v for v in male_ratio],
    name="👨 남성 (%)",
    orientation='h',
    marker=dict(color='rgba(54, 162, 235, 0.8)')
))
fig_pyramid.add_trace(go.Bar(
    y=ages,
    x=female_ratio,
    name="👩 여성 (%)",
    orientation='h',
    marker=dict(color='rgba(255, 99, 132, 0.8)')
))

fig_pyramid.update_layout(
    title=dict(text=f"📊 {selected_region} 연령별 인구 피라미드 (비율 기준)", font=dict(size=24)),
    barmode='overlay',
    xaxis=dict(title='인구 비율 (%)', tickvals=[-10, -5, 0, 5, 10], ticktext=['10%', '5%', '0', '5%', '10%']),
    yaxis=dict(title='연령'),
    height=650,
    legend=dict(x=0.02, y=1.05, orientation="h")
)

st.plotly_chart(fig_pyramid, use_container_width=True)

# 📈 전체 인구 흐름 그래프
population_total = [m + f for m, f in zip(population_male, population_female)]
df_all = pd.DataFrame({"연령": ages, "총인구": population_total})

fig_all = go.Figure(go.Bar(
    x=df_all["연령"],
    y=df_all["총인구"],
    marker=dict(color='mediumseagreen'),
    text=df_all["총인구"],
    textposition='outside',
    hovertemplate='연령 %{x}<br>인구수 %{y:,}명<extra></extra>'
))

fig_all.update_layout(
    title=dict(text="📈 전체 연령대별 인구 분포", font=dict(size=24)),
    xaxis_title="연령",
    yaxis_title="인구 수",
    height=500,
    margin=dict(t=60, l=60, r=40, b=40)
)

st.plotly_chart(fig_all, use_container_width=True)

# 🔍 유사한 지역 찾기 (동 단위, 혼합 기준: 비율 + 절댓값 차이 포함)
def hybrid_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    ratio_dist = np.linalg.norm((vec1 / vec1.sum()) - (vec2 / vec2.sum()))
    scale_dist = abs(vec1.sum() - vec2.sum()) / vec1.sum()
    return ratio_dist + scale_dist

current_vector = np.array(population_total)

best_match = None
best_score = float('inf')

for _, row in df_gender.iterrows():
    if row["지역명"] == selected_region:
        continue
    male = row[age_columns_male].str.replace(",", "").fillna("0").astype(int).tolist()
    female = row[age_columns_female].str.replace(",", "").fillna("0").astype(int).tolist()
    total_vec = np.array([m + f for m, f in zip(male, female)])
    if total_vec.sum() == 0:
        continue
    score = hybrid_distance(current_vector, total_vec)
    if score < best_score:
        best_score = score
        best_match = row["지역명"]
        best_total = total_vec.tolist()

# 📊 선택 지역 인구 구조 분석

def extract_age(age_label):
    if '이상' in age_label:
        return 100
    return int(age_label.replace('세', '').strip())

age_ranges = list(range(len(ages)))
under20 = [i for i in age_ranges if extract_age(ages[i]) < 20]
youth = [i for i in age_ranges if 20 <= extract_age(ages[i]) < 40]
middle = [i for i in age_ranges if 40 <= extract_age(ages[i]) < 65]
elderly = [i for i in age_ranges if extract_age(ages[i]) >= 65]

total_population = sum(population_total)
under20_ratio = round(sum(population_total[i] for i in under20) / total_population * 100, 2)
youth_ratio = round(sum(population_total[i] for i in youth) / total_population * 100, 2)
middle_ratio = round(sum(population_total[i] for i in middle) / total_population * 100, 2)
elderly_ratio = round(sum(population_total[i] for i in elderly) / total_population * 100, 2)

st.markdown(f"""
### 🧾 {selected_region} 인구 비율 분석
- 전체 인구: **{total_population:,}명**
- 👶 0~19세 (어린이·청소년): **{under20_ratio}%**
- 👩‍🎓 20~39세 (청년): **{youth_ratio}%**
- 👨‍💼 40~64세 (중장년): **{middle_ratio}%**
- 🧓 65세 이상 (고령): **{elderly_ratio}%**
""")

summary = "📌 인구 분석 요약: "
st.write("")  # 시각적 여백

st.markdown("### 🧠 지역별 인구 구조에 따른 종합 분석")

insights = []

if under20_ratio >= 20:
    insights.append(
        "👶 **어린이·청소년 비중이 높은 지역입니다.**  \n"
        "- 학군, 놀이시설, 방과후 돌봄센터, 청소년 문화공간이 필요합니다.  \n"
        "- 학원가, 문구점, 키즈카페 중심의 상권이 형성될 수 있습니다."
    )

if youth_ratio >= 30:
    insights.append(
        "👩‍🎓 **청년층이 많은 지역입니다.**  \n"
        "- 청년 주거, 창업 공간, 문화 예술 공간 수요가 큽니다.  \n"
        "- 공유오피스, 감성 카페, 푸드트럭 거리 등이 어울립니다."
    )

if middle_ratio >= 35:
    insights.append(
        "👨‍💼 **중장년층 비중이 높은 지역입니다.**  \n"
        "- 평생교육센터, 건강검진센터, 재취업 지원시설이 요구됩니다.  \n"
        "- 약국, 대형마트 중심의 실속형 상권이 효과적입니다."
    )

if elderly_ratio >= 25:
    insights.append(
        "🧓 **고령 인구가 많은 지역입니다.**  \n"
        "- 복지센터, 실버문화센터, 무장애 인프라 구축이 중요합니다.  \n"
        "- 전통시장이나 의료 접근성이 뛰어난 상권이 적합합니다."
    )

if not insights:
    insights.append(
        "🏙️ **세대가 균형 있게 분포한 지역입니다.**  \n"
        "- 가족 단위 복합문화시설, 도서관, 커뮤니티센터가 적합합니다.  \n"
        "- 세대 연계를 고려한 복합형 상권이 유리합니다."
    )

for insight in insights:
    st.markdown(insight)
    st.write("")  # 인사이트 간 줄바꿈

# 📌 인구 분석 요약
summary_lines = []

if under20_ratio >= 20:
    summary_lines.append(
        f"🧒 이 지역은 교복 입은 학생들과 아이들 웃음소리가 끊이지 않는 동네입니다. "
        f"{under20_ratio}%에 달하는 학령 인구는 미래를 위한 교육 인프라가 필요함을 시사합니다."
    )

if youth_ratio >= 30:
    summary_lines.append(
        f"🧑‍🎓 청년이 많은 이곳은 활력과 가능성의 중심지입니다. "
        f"전체 인구의 {youth_ratio}%를 차지하는 청년층은 일자리, 주거, 문화 공간을 갈망합니다."
    )

if middle_ratio >= 35:
    summary_lines.append(
        f"👨‍👩‍👧‍👦 중장년층이 주를 이루는 안정된 지역입니다. "
        f"{middle_ratio}%에 이르는 이들의 삶의 질을 높이려면 건강관리, 평생교육, 커뮤니티 공간이 뒷받침되어야 합니다."
    )

if elderly_ratio >= 25:
    summary_lines.append(
        f"🧓 인생의 후반전을 살아가는 어르신들이 눈에 띄는 지역입니다. "
        f"{elderly_ratio}%에 달하는 고령층은 복지시설과 접근성 좋은 환경을 요구합니다."
    )

if not summary_lines:
    summary_lines.append(
        "🏙️ 이 지역은 어린이부터 어르신까지 다양한 세대가 어울려 사는 균형 잡힌 동네입니다. "
        "세대 간 조화를 위한 다세대 복합 공간이 잘 어울릴 것입니다."
    )
elif len(summary_lines) >= 2:
    first = summary_lines[0]
    rest = [f"더불어 {line}" for line in summary_lines[1:]]
    summary_lines = [first] + rest
    
st.markdown("#### 🧾 인구 분석 요약")  # 제목 표시
st.info("\n\n".join(summary_lines))   # 강조된 요약 박스 출력
#st.markdown("  \n\n".join(summary_lines))
st.write("")

# 📍 유사 지역 시각화 (겹쳐서 비교)
st.markdown(f"### 🔄 {selected_region} 와(과) 가장 유사한 동: **{best_match}**")

fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(
    x=ages,
    y=population_total,
    mode='lines+markers',
    name=selected_region,
    line=dict(color='royalblue')
))
fig_compare.add_trace(go.Scatter(
    x=ages,
    y=best_total,
    mode='lines+markers',
    name=best_match,
    line=dict(color='orangered', dash='dot')
))

fig_compare.update_layout(
    title="👥 선택 동과 유사 동의 연령별 인구 구조 비교",
    xaxis_title="연령",
    yaxis_title="인구 수",
    height=500,
    legend=dict(x=0.01, y=1.1, orientation="h")
)

st.plotly_chart(fig_compare, use_container_width=True)
