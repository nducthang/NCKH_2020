# xử lý dữ liệu
import re

dict = [["ship", "vận chuyển"], ["shop", "cửa hàng"], ["m", "mình"], ["mik", "mình"], ["ko", "không"], ["k", "không"],
        ["kh", "không"], ["khong", "không"], ["kg", "không"], ["khg", "không"], ["tl", "trả lời"],
        ["rep", "trả lời"], ["r", "rồi"], ["fb", "facebook"], ["face", "faceook"], ["thanks", "cảm ơn"],
        ["thank", "cảm ơn"], ["tks", "cảm ơn"], ["tk", "cảm ơn"], ["ok", "tốt"], ["oki", "tốt"], ["okie", "tốt"],
        ["sp", "sản phẩm"],
        ["dc", "được"], ["vs", "với"], ["đt", "điện thoại"], ["thjk", "thích"], ["thik", "thích"], ["qá", "quá"],
        ["trể", "trễ"], ["bgjo", "bao giờ"], ["h", "giờ"], ["qa", "quá"], ["dep", "đẹp"], ["xau", "xấu"],
        ["ib", "nhắn tin"],
        ["cute", "dễ thương"], ["sz", "size"], ["good", "tốt"], ["god", "tốt"], ["bt", "bình thường"], ["o", "không"],
        ["l", "lắm"]]


# xóa các từ bị kéo dài
def remove(text):
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
    return text


# chuyển các ký tự viết hoa sang ký tự viết thường
def A_cvt_a(text):
    text = text.lower()
    return text


# Thay thế các từ viết tắt
def utils_data(text):
    list_text = text.split(" ")
    for i in range(len(list_text)):
        for j in range(len(dict)):
            if list_text[i] == dict[j][0]:
                list_text[i] = dict[j][1]
    text = " ".join(list_text)
    return text


# Đưa ra dữ liệu chuẩn
def text_util_final(text):
    text = remove(text)
    text = A_cvt_a(text)
    text = utils_data(text)
    return text


if __name__ == '__main__':
    text = "Hàng quá đẹp shop ship hàng nhanh rep ib Nhiệt tình giá rẻ 1 nửa so vs thị trường sẽ ủng hộ shop dài dai " \
           "ib " \
           "thanks shop nhiều nhaaaaa "
    text = text_util_final(text)
    print(text)
