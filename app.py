
import streamlit as st
import pandas as pd
import pickle
 
st.title("Akıllı Telefon Bağımlılığı Tahmini :iphone:")
model=pickle.load(open('telefon.pkl','rb'))
kolonlar=pickle.load(open('kolonlar.pkl','rb'))
 
yas=st.number_input('Yaş',18,35,25)
ekran=st.number_input('Günlük ekran süresi (saat)',0.5,15.0,7.5)
hafta_sonu=st.number_input('Hafta sonu ekran süresi (saat)',0.5,18.0,9.5)
sosyal=st.number_input('Sosyal medya (saat)',0.0,8.0,2.5)
oyun=st.number_input('Oyun (saat)',0.0,4.0,1.5)
is_ders=st.number_input('İş / ders (saat)',0.0,6.0,2.5)
uyku=st.number_input('Uyku (saat)',4.5,9.0,7.0)
bildirim=st.number_input('Günlük bildirim sayısı',20,250,150)
acilis=st.number_input('Günlük uygulama açma sayısı',15,180,100)
cinsiyet=st.radio('Cinsiyet',['Female','Male','Other'])
stres=st.selectbox('Stres seviyesi',['Low','Medium','High'])
etki=st.radio('Okul / iş hayatına etkisi var mı?',['Yes','No'])
 
if st.button('Tahmin et'):
    veri={'age':yas,'daily_screen_time_hours':ekran,'social_media_hours':sosyal,'gaming_hours':oyun,'work_study_hours':is_ders,'sleep_hours':uyku,'notifications_per_day':bildirim,'app_opens_per_day':acilis,'weekend_screen_time':hafta_sonu,'gender':cinsiyet,'stress_level':stres,'academic_work_impact':etki}
    x=pd.get_dummies(pd.DataFrame([veri]))
    x=x.reindex(columns=kolonlar,fill_value=0)
    tahmin=model.predict_proba(x)[0][1]
    st.success(f'Bağımlılık olasılığı: %{tahmin*100:.1f}')
 