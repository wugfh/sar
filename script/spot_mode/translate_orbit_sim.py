import csv
import re
import sys

def convert_txt_to_csv(input_path: str, output_path: str):
    """
    通过检索连续空格（2个及以上）对数据行分列，转换为 CSV。
    适用于时间字段内部有单空格、列间有多个空格的格式。
    """
    headers = ["Time (UTCG)", "Latitude (deg)", "Longitude (deg)", "Range (km)"]

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 找到表头行，以确定数据起始位置
    data_start = None
    for i, line in enumerate(lines):
        if "Time (UTCG)" in line and "Latitude" in line:
            data_start = i + 2   # 跳过表头下的分隔线
            break

    if data_start is None:
        raise ValueError("未找到数据表头，请检查文件格式。")

    rows = []
    for line in lines[data_start:]:
        line = line.rstrip('\n')
        if not line.strip():
            continue   # 跳过空行

        # 使用连续空格（2个及以上）作为分隔符进行分割
        parts = re.split(r'\s{2,}', line)
        # 去除可能由行首空格产生的空字符串
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) != 4:
            raise ValueError(f"分列结果不为4列，请检查行：{line}")

        rows.append(parts)

    # 写入 CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"转换完成，共 {len(rows)} 行数据，输出至 {output_path}")

if __name__ == "__main__":
    convert_txt_to_csv("E:/Download/Sensor1_Boresight_Intersection.txt", "E:/Download/Sensor1_Boresight_Intersection.csv")