import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import iplot



@st.cache
def load_data(op):
    if op=='dropout':
        dropout = pd.read_csv('D:\Indian-School-Management-Statistics-Analysis\datasets\dropout-ratio.csv')
        return dropout
    elif op=='enrollment':
        enroll = pd.read_csv('D:\Indian-School-Management-Statistics-Analysis\datasets\enrollment-ratio.csv')
        return enroll
    elif op=='computers':
        comps = pd.read_csv('D:\Indian-School-Management-Statistics-Analysis\datasets\percentage-of-schools-with-comps.csv')
        return comps
    elif op=='boys toilet':
        boys_toilet = pd.read_csv('D:\Indian-School-Management-Statistics-Analysis\datasets\schools-with-boys-toilet.csv')
        return boys_toilet
    elif op=='girls toilet':
        girls_toilet = pd.read_csv('D:\Indian-School-Management-Statistics-Analysis\datasets\schools-with-girls-toilet.csv')
        return girls_toilet

st.title("School Management statistics analysis")
st.subheader("This is a streamlit dashboard to analyze school management statistics")

def dropout_analysis():   
    df=load_data('dropout')
    dropout1 = df[['State_UT','year','Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total']]
    c = dropout1.loc[dropout1['year'] =='2012-13']
    c = c.replace(to_replace='NR',value='NaN',regex = True)
    c['Primary_Total'] = c['Primary_Total'].values.astype(np.float32)
    c['Upper Primary_Total'] = c['Upper Primary_Total'].values.astype(np.float32)
    c['Secondary _Total'] = c['Secondary _Total'].values.astype(np.float32)
    c['HrSecondary_Total'] = c['HrSecondary_Total'].values.astype(np.float32)
    years=df.year.unique()
    sel_year = st.selectbox("Select Year",years)
    if sel_year==years[0]:
        fig, ax = plt.subplots(figsize =(15,5))
        ax.set_facecolor('black')
        plt.xticks(rotation='vertical')
        plt.bar(c['State_UT'],c['Primary_Total'])
        plt.bar(c['State_UT'],c['Upper Primary_Total'],color = 'Black')
        plt.bar(c['State_UT'],c['Secondary _Total'])
        plt.bar(c['State_UT'],c['HrSecondary_Total'])
        plt.title('2012-13')
        plt.legend(['Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total'])
        st.pyplot(fig)
    elif sel_year==years[1]:
        fig,ax = plt.subplots(figsize = (15,5))
        ax.set_facecolor('black')
        plt.xticks(rotation = 'vertical')
        plt.bar(c['State_UT'],c['Primary_Total'])
        plt.bar(c['State_UT'],c['Upper Primary_Total'],color = 'yellow')
        plt.bar(c['State_UT'],c['Secondary _Total'])
        plt.bar(c['State_UT'],c['HrSecondary_Total'], color = 'pink')
        plt.title('2013-14')
        plt.legend(['Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total'])
        st.pyplot(fig)
    elif sel_year==years[2]:
        fig,ax = plt.subplots(figsize = (15,5))
        ax.set_facecolor('black')
        plt.xticks(rotation = 'vertical')
        plt.bar(c['State_UT'],c['Primary_Total'])
        plt.bar(c['State_UT'],c['Upper Primary_Total'],color = 'red')
        plt.bar(c['State_UT'],c['Secondary _Total'])
        plt.bar(c['State_UT'],c['HrSecondary_Total'], color = 'cyan')
        plt.title('2014-15')
        plt.legend(['Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total'])
        st.pyplot(fig)

def enrollment_analysis():
    df=load_data('enrollment')
    df = df.replace(to_replace='NR',value = 'NaN', regex = True)
    df = df.replace(to_replace='@', value = 'NaN', regex = True)
    df['Higher_Secondary_Boys'] = df['Primary_Total'].values.astype(np.float64)
    df['Higher_Secondary_Girls'] = df['Higher_Secondary_Girls'].values.astype(np.float64)
    df['Higher_Secondary_Total'] = df['Higher_Secondary_Total'].values.astype(np.float64)
    df1 = df.loc[df['State_UT']== 'All India']
    f = df.loc[df['Year'] == '2013-14']
    g = df.loc[df['Year'] == '2014-15']
    h = df.loc[df['Year'] == '2015-16']
    def boys_to_girls():
        ig, ax = plt.subplots(figsize =(15,5))
        X = np.arange(3)
        plt.bar(X + 0.00, df1['Primary_Boys'], color = 'b', width = 0.05, label='Primary_Boys')
        plt.bar(X + 0.05, df1['Primary_Girls'], color = 'g', width = 0.05,label = 'U_Primary_Girls')
        plt.bar(X + 0.15, df1['Upper_Primary_Boys'], color = 'r', width = 0.05,label='Upper_Primary_Boys')
        plt.bar(X + 0.20, df1['Upper_Primary_Girls'], color = 'm', width = 0.05,label='Upper_Primary_Girls')
        plt.bar(X + 0.30, df1['Secondary_Boys'], color = 'c', width = 0.05,label='Secondary_Boys')
        plt.bar(X + 0.35, df1['Secondary_Girls'], color = 'y', width = 0.05,label='Secondary_Girls')
        plt.bar(X + 0.45, df1['Higher_Secondary_Boys'], color = 'k', width = 0.05,label='Higher_Secondary_Boys')
        plt.bar(X + 0.50, df1['Higher_Secondary_Girls'], color = 'grey', width = 0.05,label='Higher_Secondary_Girls')
        plt.legend(['Primary_Boys','Primary_Girls','Upper_Primary_Boys','Upper_Primary_Girls','Secondary_Boys','Secondary_Girls',
                    'Higher_Secondary_Boys','Higher_Secondary_Girls'], loc='center left',bbox_to_anchor=(1, 0.5))
        plt.xlabel('YEAR', fontweight ='bold') 
        plt.ylabel('PERCENTAGE', fontweight ='bold') 
        plt.xticks([r + 0.25 for r in range(0,3)], 
                ['2013-14', '2014-15', '2015-16']) 
        plt.title('All India Boys & Girls Enrollment Ratio')
        st.pyplot(ig)
    def all_india():
        X = np.arange(3)
        fig, ax = plt.subplots(figsize =(15,5)) 
        plt.bar(X + 0.00, df1['Primary_Total'], color = 'b', width = 0.10, label='Primary_Total')
        plt.bar(X + 0.10, df1['Upper_Primary_Total'], color = 'g', width = 0.10,label = 'Upper_Primary_Total')
        plt.bar(X + 0.20, df1['Secondary_Total'], color = 'r', width = 0.10,label='Secondary_Total')
        plt.bar(X + 0.30, df1['Higher_Secondary_Total'], color = 'm', width = 0.10,label='Higher_Secondary_Total')
        plt.legend(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'], loc='center left',bbox_to_anchor=(1, 0.5))
        plt.xlabel('YEAR', fontweight ='bold') 
        plt.ylabel('PERCENTAGE', fontweight ='bold') 
        plt.xticks([r + 0.20 for r in range(0,3)], 
                ['2013-14', '2014-15', '2015-16']) 
        plt.title('All India Enrollment Ratio')
        st.pyplot(fig)
    ops=['Boys to girls ratio', 'Class wise ratio','Year wise ratio']
    ch=st.selectbox('Select an option',ops)
    if ch==ops[0]:
        boys_to_girls()
    elif ch==ops[1]:
        all_india()
    elif ch==ops[2]:
        o=['2013-14','2014-15','2015-16']
        s=st.selectbox('Select',o)
        if s==o[0]:
            fig,ax = plt.subplots(figsize =(15,5))
            plt.xticks(rotation='vertical')
            plt.bar(f['State_UT'],f['Primary_Total'])
            plt.bar(f['State_UT'],f['Upper_Primary_Total'])
            plt.bar(f['State_UT'],f['Secondary_Total'])
            plt.bar(f['State_UT'],f['Higher_Secondary_Total'])
            plt.title('2013-14')
            plt.legend(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'])
            st.pyplot(fig)
        elif s==o[1]:
            fig, ax = plt.subplots(figsize =(15,5))
            plt.xticks(rotation='vertical')
            plt.bar(g['State_UT'],g['Primary_Total'])
            plt.bar(g['State_UT'],g['Upper_Primary_Total'])
            plt.bar(g['State_UT'],g['Secondary_Total'])
            plt.bar(g['State_UT'],g['Higher_Secondary_Total'])
            plt.title('2014-15')
            plt.legend(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'])
            st.pyplot(fig)
        elif s==o[2]:
            fig,ax = plt.subplots(figsize =(15,5))
            plt.xticks(rotation='vertical')
            plt.bar(h['State_UT'],h['Primary_Total'])
            plt.bar(h['State_UT'],h['Upper_Primary_Total'])
            plt.bar(h['State_UT'],h['Secondary_Total'])
            plt.bar(h['State_UT'],h['Higher_Secondary_Total'])
            plt.title('2015-16')
            plt.legend(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'])
            st.pyplot(fig)

def schools_with_comp():
    df=load_data('computers')
    list1 = df.sort_values(['All Schools'], ascending=False)
    ops=['Visualizing computerized contribution of various states','Trend of computer in schools from various states']
    sel = st.selectbox("Select Year",ops)
    if sel==ops[0]:
        o=['Using bar chart','Using pie chart']
        s=st.selectbox('Select',o)
        if s==o[0]:
            a=px.bar(data_frame=list1,x = 'State_UT', y = 'All Schools', 
                            labels={'x':'State and UT', 'y':'All Schools'},
                            opacity=1,color_discrete_sequence=['red'])
            st.plotly_chart(a)
        if s==o[1]:
            chart = px.pie(data_frame=list1,values='All Schools',names='State_UT',height=600)
            chart.update_traces(textposition='inside',textinfo = 'percent+label')
            chart.update_layout(title_x = 0.5, geo = dict(showframe = False,showcoastlines= False))
            st.plotly_chart(chart)
    if sel==ops[1]:
        comp1 = df.copy()
        x = comp1.State_UT
        trace_1 = {
            'x':x,
            'y':comp1.Primary_Only,
            'name':'Primary_Education',
            'type':'bar'
        };
        trace_2 = {
            'x':x,
            'y':comp1.Sec_Only,
            'name':'comp1.Secondary_Education',
            'type':'bar'
        };
        trace_3 = {
            'x':x,
            'y':comp1.HrSec_Only,
            'name':'HigherSecondary',
            'type':'bar',
        };
        trace_4 = {
            'x':x,
            'y':comp1.U_Primary_Only,
            'name':'UnderPrimary',
            'type':'bar',
        };
        data = [trace_1,trace_2,trace_3,trace_4]
        layout = {
            'xaxis':{'title':'Presence of Computers in Education'},
            'barmode':'relative',
            'title':'Trend Of computing Education in Indian States',
        }
        fig = go.Figure(data = data,layout=layout)
        st.plotly_chart(fig)

def boys_toilet():
    df=load_data('boys toilet')
    boys_melt = pd.melt(df, id_vars=['State_UT', 'year'], var_name='School_Level', value_name = 'toilet')
    ops=['Year-wise percentage','State-wise percentage']
    s_ops=st.selectbox('Select',ops)
    if s_ops==ops[0]:
        years=['2013-14','2014-15','2015-16']
        sel=st.selectbox('Select Year',years)
        if sel==years[0]:
            boys_2013 = boys_melt.iloc[np.where(boys_melt.year=='2013-14')]
            fig=boys_2013.groupby(['School_Level']).mean().sort_values(by='toilet')
            st.subheader('Toilet Failities for boys in all School Categories in 2013-14 session')
            st.bar_chart(fig,height=500)
        elif sel==years[1]:
            boys_2014 = boys_melt.iloc[np.where(boys_melt.year=='2014-15')]
            fig=boys_2014.groupby(['School_Level']).mean().sort_values(by='toilet')
            st.subheader('Toilet Failities for boys in all School Categories in 2014-15 session')
            st.bar_chart(fig,height=500)
        elif sel==years[2]:
            boys_2015 = boys_melt.iloc[np.where(boys_melt.year=='2015-16')]
            fig=boys_2015.groupby(['School_Level']).mean().sort_values(by='toilet')
            st.subheader('Toilet Failities for boys in all School Categories in 2015-16 session')
            st.bar_chart(fig,height=500)
    if s_ops==ops[1]:
        fig=boys_melt.groupby(['State_UT'])['toilet'].mean().sort_values()
        st.bar_chart(fig,height=500, use_container_width=False)

def girs_toilet():
    df=load_data('girls toilet')
    girls_melt = pd.melt(df, id_vars=['State_UT', 'year'], var_name='School_Level', value_name = 'toilet')
    ops=['Year-wise percentage','State-wise percentage']
    s_ops=st.selectbox('Select',ops)
    if s_ops==ops[0]:
        years=['2013-14','2014-15','2015-16']
        sel=st.selectbox('Select Year',years)
        if sel==years[0]:
            girls_2013 = girls_melt.iloc[np.where(girls_melt.year=='2013-14')]
            fig=girls_2013.groupby(['School_Level']).mean().sort_values(by='toilet')
            st.subheader('Toilet Failities for girls in all School Categories in 2013-14 session')
            st.bar_chart(fig,height=500)
        elif sel==years[1]:
            girls_2014 = girls_melt.iloc[np.where(girls_melt.year=='2014-15')]
            fig=girls_2014.groupby(['School_Level']).mean().sort_values(by='toilet')
            st.subheader('Toilet Failities for girls in all School Categories in 2014-15 session')
            st.bar_chart(fig,height=500)
        elif sel==years[2]:
            girls_2015 = girls_melt.iloc[np.where(girls_melt.year=='2015-16')]
            fig=girls_2015.groupby(['School_Level']).mean().sort_values(by='toilet')
            st.subheader('Toilet Failities for girls in all School Categories in 2015-16 session')
            st.bar_chart(fig,height=500)
    if s_ops==ops[1]:
        fig=girls_melt.groupby(['State_UT'])['toilet'].mean().sort_values()
        st.bar_chart(fig,height=500, use_container_width=False)

options = ['Dropout','Enrollment','Schools with Computer','Boys Toilet facility','Girls Toilet facility']
choice = st.sidebar.radio("Select any option", options)
if choice == options[0]:
    dropout_analysis()
elif choice == options[1]:
    enrollment_analysis()
elif choice == options[2]:
    schools_with_comp()
elif choice == options[3]:
    boys_toilet()
elif choice == options[4]:
    girs_toilet()