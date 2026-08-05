import streamlit as st
import requests
import time

st.set_page_config(
    page_title="AI Feedback Analysis",
    layout="centered"
)

if "ai_ready" not in st.session_state:
    st.session_state.ai_ready = False

st.title("Hệ thống Phân tích Phản hồi Sinh viên", width="content", text_alignment="center")
st.markdown("Nhập đánh giá của bạn về môn học, giảng viên hoặc cơ sở vật chất. AI sẽ phân tích nó ngay lập tức!")

user_input = st.text_area(
    "Nội dung phản hồi:", 
    placeholder="Ví dụ: Thầy giảng bài rất hay, nhưng phòng máy lạnh bị hỏng nóng quá...",
    height=150
)

API_HEALTH_URL = "http://localhost:8000/health"
API_PREDICT_URL = "http://localhost:8000/predict"

button_placeholder = st.empty()
result_container = st.container()

if not st.session_state.ai_ready:
    try:
        res = requests.get(API_HEALTH_URL, timeout=1)
        if res.status_code == 200:
            st.session_state.ai_ready = True
    except requests.exceptions.ConnectionError:
        pass

# ==========================================
# GIAO DIỆN KHI AI CHƯA SẴN SÀNG
# ==========================================
if not st.session_state.ai_ready:
    button_placeholder.button("Đang nạp model AI", disabled=True, use_container_width=True)
    
    for _ in range(10):
        time.sleep(1)
        try:
            res = requests.get(API_HEALTH_URL, timeout=1)
            if res.status_code == 200:
                st.session_state.ai_ready = True
                st.rerun()
        except requests.exceptions.ConnectionError:
            pass

    if not st.session_state.ai_ready:
        st.error("🚨 Không thể kết nối đến AI Server. Vui lòng kiểm tra lại!")
        st.stop()

# ==========================================
# GIAO DIỆN KHI AI ĐÃ SẴN SÀNG
# ==========================================
if st.session_state.ai_ready:
    if button_placeholder.button("Phân tích bằng AI", use_container_width=True):
        if user_input.strip():
            with result_container:
                with st.spinner("AI đang xử lý ngôn ngữ..."):
                    try:
                        response = requests.post(API_PREDICT_URL, json={"text": user_input})
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    label="📌 Chủ đề (Topic)", 
                                    value=result['topic']['label'], 
                                    delta=f"Độ tin cậy: {result['topic']['confidence']}%",
                                    delta_color="off"
                                )
                                
                            with col2:
                                sentiment = result['sentiment']['label']
                                delta_color = "normal" if sentiment == "Tích cực" else "inverse" if sentiment == "Tiêu cực" else "off"
                                st.metric(
                                    label="🎭 Cảm xúc (Sentiment)", 
                                    value=sentiment, 
                                    delta=f"Độ tin cậy: {result['sentiment']['confidence']}%",
                                    delta_color=delta_color
                                )
                                
                            with st.expander("Xem chi tiết văn bản sau khi chuẩn hóa"):
                                st.code(result['cleaned_text'], language="text")
                        else:
                            st.error(f"⚠️ Lỗi từ Server API: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Mất kết nối đột ngột: {str(e)}")
        else:
            result_container.warning("Bạn chưa nhập nội dung gì cả!")