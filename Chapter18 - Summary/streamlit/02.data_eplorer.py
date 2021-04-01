import streamlit as st
import numpy as np
import pandas as pd

st.title("Tạo ứng dụng khám phá dữ liệu")
st.write("Trong phần này, chúng ta sẽ sử dụng các tính năng cốt lõi của Streamlit để tạo một ứng dụng tương tác, khám phá tập dữ liệu Uber công khai về các điểm đón và trả khách ở thành phố New York. Khi hoàn tất, bạn sẽ biết cách tìm nạp và lưu dữ liệu vào bộ nhớ cache, vẽ biểu đồ, vẽ thông tin trên bản đồ và sử dụng các tiện ích tương tác, như thanh trượt, thanh lọc kết quả.")
st.write("Đầu tiên chúng ta cần import một số thư viện cần thiết")
with st.echo():
    import streamlit as st
    import numpy as np
    import pandas as pd

st.title("Fetch some data")
with st.echo():
    DATE_COLUMN = 'date/time'
    DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
    def load_data(nrows):
        ''' nrows is number of row that you want to load'''
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    # Create a text element and let the reader know data is loading
    data_load_state = st.text('Loading data...')
    # Load 10000 rows of data into the dataframe
    data = load_data(10000)
    # Notify the reader that the data was successfully loader
    data_load_state.text('Loading data... done!')


# Effortless caching
st.title("Sử dụng cache (bộ nhớ đệm)")
st.write("1. Chúng ta thêm **@st.cache** trước hàm **load_data**. Sau đó, lưu tập lệnh và Streamlit sẽ tự động chạy lại ứng dụng của bạn.  Vì đây là lần đầu tiên bạn chạy tập lệnh với @ st.cache, bạn sẽ không thấy bất kỳ điều gì thay đổi.  Hãy chỉnh sửa tệp của bạn thêm một chút để bạn có thể thấy sức mạnh của bộ nhớ đệm.")
st.write("2. Thay dòng **data_load_state.text('Loading data... done!')** thành:")
st.write(">data_load_state.text('Done! (using st.cache)")
st.write("4. Tốc độ nhanh hẳn vì đã lưu cache")

with st.echo():
    DATE_COLUMN = 'date/time'
    DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
    
    @ st.cache()
    def load_data(nrows):
        ''' nrows is number of row that you want to load'''
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    # Create a text element and let the reader know data is loading
    data_load_state = st.text('Loading data...')
    # Load 10000 rows of data into the dataframe
    data = load_data(10000)
    # Notify the reader that the data was successfully loader
    data_load_state.text('Done! (using st.cache)')

st.title("Kiểm tra dữ liệu thô")
with st.echo():
    st.write(data)


st.title("Vẽ biểu đồ")
with st.echo():
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)

st.write("Xem thêm các phương thức hiển thị chart khác tại: https://docs.streamlit.io/en/stable/api.html#display-charts")

st.title("Vẽ dữ liệu trên map")
with st.echo():
    st.map(data)

st.title("Lọc kết quả với slider")
with st.echo():
    hour_to_filter = st.slider('hour', 0, 23, 17)   # min: 0h, max: 23h, default: 17h
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader(f'Map of all pickups at {hour_to_filter}:00')
    st.map(filtered_data)