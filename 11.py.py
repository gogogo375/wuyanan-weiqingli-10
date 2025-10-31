import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
model=joblib.load("score_predictor.pkl")
st.set_page_config(page_title='💯学生成绩分析平台',layout='wide')
def introduce_page ():
    """当选择项目介绍页面时，将呈现该函数的内容""" 
    st.title("💯学生成绩分析与预测系统") 
    st.markdown('***')
    c1,c2 = st.columns([2,1])
    with c1:
        st.header("📝项目概述")
        st.text("本项目是基于Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。") 
        st.subheader("主要特点")
        st.text(" ·📊数据可视化:多维度展示学生学业数据。")
        st.text(" ·🎯专业分析:按专业分类的详细统计分析。")
        st.text(" ·🤔智能预测:基于机器学习模型的成绩预测。")
        st.text(" ·💡学习建议:根据预测结果提供个性化反馈。")
    with c2:
        images = [
    {'url':'data/1.png' },
    {'url':'data/2.png'}]
        if'ind'not in st.session_state:
            st.session_state['ind']=0


        def nextImg():
            st.session_state['ind']=(st.session_state['ind']-1)%len(images)
        def nextimg():
            st.session_state['ind']=(st.session_state['ind']+1)%len(images)
        st.image(images[st.session_state['ind']]['url'])


        h1,h2 = st.columns(2)
        with h1:
            st.button('上一张',on_click=nextImg,use_container_width=True)
        with h2:
            st.button('下一张',on_click=nextimg,use_container_width=True)


    
    st.markdown('***')
    
    st.header("🚀项目目标")
    a1,a2,a3= st.columns([1,1,1])
    with a1:
        st.subheader("🎯目标一")
        st.text("分析影响因素")
        st.text(" · 识别关键学习指标")
        st.text(" · 探索成绩相关因素")
        st.text(" · 提供数据支持决策")
    with a2:
        st.subheader("📈目标二")
        st.text("可视化展示")
        st.text(" · 专业对比分析")
        st.text(" · 性别差异研究")
        st.text(" · 学习模式识别")
    with a3:
        st.subheader("🔮目标三")
        st.text("成绩预测")
        st.text(" · 机器学习模型")
        st.text(" · 个性化预测")
        st.text(" · 及时干预预警")
    st.markdown('***')
    st.header("🛠技术架构")
    b1,b2,b3,b4= st.columns([1,1,1,1])
    with b1:
        st.text("前端框架")
        st.markdown('Streamlit')
        
    with b2:
        
        st.text("数据处理")
        st.markdown('Pandas')
        st.markdown('NumPy')
     
    with b3:
        
        st.text("可视化")
        st.markdown('Plotly')
        st.markdown('Matpotlib')
    with b4:
        st.text("机器学习")
        st.markdown("Scikit_learn")
      



def second_page():
    """当选择专业数据分析页面时，将呈现该函数的内容"""
    st.header("📊专业数据分析")
    st.markdown('***')
    st.subheader("1.各专业男女性别比例")
    st.text("各专业男女性别比例")
    pd.set_option ('display.unicode.east_asian_width', True) 
    df = pd. read_csv('data/student_data_adjusted_rounded.csv',encoding='utf-8') 
    c3, c4 = st.columns([2, 1])

    with c3:
        gender_fig = px.histogram(df, x="专业", color="性别", barmode="group",title="各专业男女性别比例",labels={"专业": "专业", "count": "人数"})
        st.plotly_chart(gender_fig, use_container_width=True)
    with c4:

        gender_data = df.groupby(["专业", "性别"])["学号"].count().reset_index()

        gender_pivot = gender_data.pivot(index="专业", columns="性别", values="学号").fillna(0)

        st.subheader("性别比例数据")

        st.dataframe(gender_pivot, use_container_width=True)

    st.markdown("***")
    st.subheader("2. 各专业学习指标对比")
    metrics = ["每周学习时长（小时）", "期中考试分数", "期末考试分数"]

    metric_df = df.groupby("专业")[metrics].mean().reset_index()

    col1, col2 = st.columns([2, 1])

    with col1:

        metric_fig = go.Figure()

        for metric in metrics:

            metric_fig.add_trace(go.Scatter(x=metric_df["专业"], y=metric_df[metric], name=metric))

        metric_fig.update_layout(title="各专业学习指标趋势对比",

                                 xaxis_title="专业", yaxis_title="指标值")

        st.plotly_chart(metric_fig, use_container_width=True)

    with col2:

        st.subheader("详细数据")

        st.dataframe(metric_df, use_container_width=True)

    

    # 分割线

    st.markdown("---")

    st.subheader("3. 各专业出勤率分析")

    col1, col2 = st.columns([2, 1])

    with col1:

        attendance_fig = px.density_heatmap(df, x="专业", y="上课出勤率",

                                           title="各专业出勤率分布",

                                           labels={"专业": "专业", "上课出勤率": "出勤率"})

        st.plotly_chart(attendance_fig, use_container_width=True)

    with col2:

        attendance_rank = df.groupby("专业")["上课出勤率"].mean().reset_index().sort_values("上课出勤率", ascending=False)

        attendance_rank["排名"] = attendance_rank["上课出勤率"].rank(ascending=False).astype(int)

        st.subheader("出勤率排名")

        st.dataframe(attendance_rank[["排名", "专业", "上课出勤率"]], use_container_width=True)

    

    # 分割线

    st.markdown("---")



    st.subheader("4. 大数据管理专业专项分析")

    bd_major = df[df["专业"] == "大数据管理"]

    # 关键指标卡片

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric("平均出勤率", f"{bd_major['上课出勤率'].mean():.1%}")

    with col2:

        st.metric("期中成绩", f"{bd_major['期中考试分数'].mean():.1f}分")

    with col3:

        st.metric("期末成绩", f"{bd_major['期末考试分数'].mean():.1f}分")

    with col4:

        st.metric("平均学习时长", f"{bd_major['每周学习时长（小时）'].mean():.1f}小时")

    # 图表

    col1, col2 = st.columns(2)

    with col1:

        bd_score_fig = px.histogram(bd_major, x="期末考试分数", title="大数据管理专业期末成绩分布")

        st.plotly_chart(bd_score_fig, use_container_width=True)

    with col2:

        bd_hours_fig = px.box(bd_major, y="每周学习时长（小时）", title="大数据管理专业学习时长分布")

        st.plotly_chart(bd_hours_fig, use_container_width=True)
def third_page():
    """当选择成绩预测页面时，将呈现该函数的内容"""
    st.header("🔮期末成绩预测")
    st.markdown('***')
    st.text("请输入学生的学习信息，系统将预测其期末成绩并提供学习建议")
    d1, d2 = st.columns(2)
    with d1:
         number = st.text_input('学号', autocomplete='number')
         st.text("性别")
         gender = st.radio(
         '性别',
         ['男', '女', '其他'],
         horizontal=True,
         label_visibility='collapsed',
)

         st.text("专业")
         def my_format_func(option):
              return f'选{option}'
         zhuanye =st.selectbox('专业',
                ['人工智能', '大数据管理', '工商管理','电子商务','财务管理'],
         label_visibility='collapsed',
)
    with d2:
         shichang= st.slider('每周学习时间（小时）', min_value=0.0, max_value=100.0,step=0.1)
         chuqinlv= st.slider('上课出勤率', min_value=0.0, max_value=1.0,step=0.01)
         qizhong= st.slider('期中考试分数', min_value=0.0, max_value=100.0,step=0.1)
         zuoye= st.slider('作业完成率', min_value=0.0, max_value=1.0,step=0.01)
    submitted = st.button("预测结果")
    if submitted:
       X=[[shichang,chuqinlv,qizhong,zuoye]]
       
       pred_score=model.predict(X)[0]
       pred_score=max(0,min(100,pred_score))
       st.subheader("📊预测结果")
       st.markdown(f"**预测期末成绩:{pred_score:.2f}分**")
       if pred_score>=80:
          st.image("https://static.vecteezy.com/system/resources/previews/013/961/431/original/congratulations-inspirational-text-on-a-multicolored-splash-of-watercolor-paint-splashes-of-rainbow-color-banner-vector.jpg")
          st.text('🎉恭喜你，成绩优秀')
       elif pred_score>=60:
          st.success("☺成绩合格，继续保持")
          st.image("https://picx.zhimg.com/v2-920894cfd6565db46e01e5e9a17a1de1_720w.jpg?source=172ae18b")
       else:
          st.warning("💪成绩待提高，建议加强学习")
          st.image("https://img.mp.sohu.com/upload/20170512/9c8b38f8be88494f94139546c2e56b23_th.png")
       
    


    
    
   
nav =st.sidebar.radio("🎓导航菜单",["项目介绍","专业数据分析","成绩预测"]) 

if nav =="项目介绍": 
    introduce_page () 
elif nav =="专业数据分析": 
    second_page ()
elif nav =="成绩预测":
    third_page()

