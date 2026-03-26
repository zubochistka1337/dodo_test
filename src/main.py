import argparse
import time
import json
from enum import Enum
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

parser = argparse.ArgumentParser(prog='DodoTest', description='Тест для додо')
parser.add_argument('-v', '--video', type=str)
args = parser.parse_args()

if args.video is None:
    raise Exception('Не предоставлен путь до видеофайла')

root_dir = Path(__file__).parent.parent
video_path = Path(args.video)
output_path = root_dir / 'output.mp4'
model_path = root_dir / 'yolo11s.pt'
csv_path = root_dir / 'output.csv'
avg_data_path = root_dir / 'avg_data.json'
if not video_path.exists():
    raise FileExistsError(f'Видеофайл недостпуен по пути: {video_path}')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class RuntimeException(Exception):
    pass


class TableState(Enum):
    free = 0
    occupied = 1


def init_videocapture() -> tuple[cv2.VideoCapture, tuple[int, int, int, int]]:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeException(f'Не удалось инициализировать чтение кадров')

    x, y, w, h = cv2.selectROI('Choose Table', frame, showCrosshair=False, fromCenter=False)
    x1, y1, x2, y2 = int(x), int(y), int(w + x), int(h + y)
    print(f'Выбранный ROI столика: x: {x}, y: {y}, w: {w}, h: {h}')
    cv2.destroyWindow('Choose Table')
    return cap, (x1, y1, x2, y2)


def draw_frame(frame: np.ndarray,
               xyxy: tuple[int, int, int, int],
               text: str,
               color: tuple[int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    bbox_text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    bbox_text_coords = (int(x1 + 2), int(y1 + bbox_text_size[1] + 2))
    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    frame = cv2.putText(frame,
                        text,
                        bbox_text_coords,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2)
    return frame


def get_ioa_person_table(table_bbox: tuple[int, int, int, int],
                         person_bbox: tuple[int, int, int, int]) -> tuple[float, tuple[int, int, int, int]]:
    # 1. Распаковка координат человека
    p_x1, p_y1, p_x2, p_y2 = person_bbox
    p_w = p_x2 - p_x1
    p_h = p_y2 - p_y1

    # 2. Вычисляем координаты "ног" (нижняя 1/3 часть рамки)
    # Нижняя граница y2 остается такой же, а верхняя y1 смещается вниз
    legs_x1 = p_x1
    legs_y1 = p_y1 + int(p_h * 0.66)  # Берем нижнюю треть (начинаем с 66% высоты)
    legs_x2 = p_x2
    legs_y2 = p_y2

    # 3. Распаковка координат стола
    t_x1, t_y1, t_x2, t_y2 = table_bbox

    # 4. Находим координаты пересечения (Intersection)
    # Это "рамка внутри рамок"
    inter_x1 = max(legs_x1, t_x1)
    inter_y1 = max(legs_y1, t_y1)
    inter_x2 = min(legs_x2, t_x2)
    inter_y2 = min(legs_y2, t_y2)

    # 5. Считаем ширину и высоту пересечения
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1

    # Если пересечения нет, ширина или высота будут отрицательными
    if inter_w <= 0 or inter_h <= 0:
        return 0.0, (legs_x1, legs_y1, legs_x2, legs_y2)

    # 6. Считаем площади
    intersection_area = inter_w * inter_h
    legs_area = (legs_x2 - legs_x1) * (legs_y2 - legs_y1)

    # 7. Возвращаем IoA (какой процент "ног" находится в зоне стола)
    if legs_area == 0:
        return 0.0, (legs_x1, legs_y1, legs_x2, legs_y2)

    return intersection_area / legs_area, (legs_x1, legs_y1, legs_x2, legs_y2)


def get_time_by_fps_rate(fps_rate: float, fps_count: int) -> int:
    time = int(fps_count // fps_rate)
    return time


def analyse_data(data: pd.DataFrame, frame_latency: list):
    data['time_to_next_event'] = data['timestamp_from_fps'].shift(-1) - data['timestamp_from_fps']
    wait_times = data[data['events'] == 'free'].copy()
    wait_times = wait_times.dropna(subset=['time_to_next_event'])

    average_wait = wait_times['time_to_next_event'].mean()
    print(f"\nСреднее время ожидания: {round(average_wait, 2)}s")
    average_latency = np.mean(frame_latency) if frame_latency else 0.0
    print(f'\nСреднее время задержки получения и обработки кадра: {round(average_latency, 2)}s')
    avg_data = {'average_wait': average_wait, 'average_latency': average_latency}

    with open(avg_data_path, 'w', encoding='utf-8') as f:
        json.dump(avg_data, f, ensure_ascii=False, indent=4)

    data.to_csv(csv_path)


def video_runtime(cap: cv2.VideoCapture,
                  writer: cv2.VideoWriter,
                  roi: tuple[int, int, int, int],
                  ret_error_limit: int = 10,
                  ioa_start_time_threshold: int = 15,
                  ioa_end_time_threshold: int = 15,
                  frame_skip_threshold: int = 20,
                  ioa_threshold: float = 0.2) -> tuple[dict, list]:
    ret_error_counts = 0
    table_roi = roi
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda')

    data = dict()
    data['events'] = ['free']
    data['timestamp_from_fps'] = [0]
    frame_latency = list()

    print(f'Доступные классы в модели: {model.names}')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS источника: {source_fps}')
    table_state = TableState.free
    ioa_start = False
    ioa_end = False
    frames_without_ioa = 0
    frames_with_ioa = 0
    frames_counter = 0
    while True:
        infer_start = time.time()
        ret, frame = cap.read()
        if not ret:
            ret_error_counts += 1
            if ret_error_counts > ret_error_limit:
                print("Видео закончилось или достигнут лимит ошибок кадра")
                break
            continue
        ret_error_counts = 0

        model_results = model.predict(frame, imgsz=960, classes=[0], verbose=False)
        if model_results:
            model_results = model_results[0]

        at_least_one_box_ioa = False
        for box in model_results.boxes:
            if not isinstance(box, Boxes):
                continue

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            name = model.names[cls]
            bbox_text = f'{name}: {round(conf, 2)}'

            ioa, legs_coords = get_ioa_person_table(table_roi, (int(x1), int(y1), int(x2), int(y2)))
            if ioa > ioa_threshold:
                if not at_least_one_box_ioa:
                    at_least_one_box_ioa = True

            frame = draw_frame(frame, (int(x1), int(y1), int(x2), int(y2)), bbox_text, (255, 0, 0))
            frame = draw_frame(frame, legs_coords, 'legs', (91, 63, 109))

        frames_counter += 1
        if at_least_one_box_ioa:
            frames_without_ioa = 0
            frames_with_ioa += 1
        else:
            frames_with_ioa = 0
            frames_without_ioa += 1

        if table_state is TableState.free:
            if ioa_start:
                if get_time_by_fps_rate(source_fps, frames_with_ioa) > ioa_start_time_threshold:
                    table_state = TableState.occupied
                    ioa_start = False
                    data['events'].append('occupied')
                    timestamp_from_fps = get_time_by_fps_rate(source_fps, frames_counter)
                    data['timestamp_from_fps'].append(timestamp_from_fps)
            elif frames_with_ioa == frame_skip_threshold:
                ioa_start = True
        elif table_state is TableState.occupied:
            if ioa_end:
                if get_time_by_fps_rate(source_fps, frames_without_ioa) > ioa_end_time_threshold:
                    table_state = TableState.free
                    ioa_end = False
                    data['events'].append('free')
                    timestamp_from_fps = get_time_by_fps_rate(source_fps, frames_counter)
                    data['timestamp_from_fps'].append(timestamp_from_fps)
            elif frames_without_ioa == frame_skip_threshold:
                ioa_end = True

        if table_state is TableState.occupied:
            frame = draw_frame(frame, table_roi, 'Table', (0, 0, 255))
        elif table_state is TableState.free:
            frame = draw_frame(frame, table_roi, 'Table', (0, 255, 0))

        infer_end = time.time() - infer_start
        frame_latency.append(infer_end)

        writer.write(frame)

        frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    writer.release()
    return data, frame_latency


cap, roi = init_videocapture()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Frame width: {frame_width}, height: {frame_height}, fps: {fps}')
writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (frame_width, frame_height))

data, frame_latency = video_runtime(cap, writer, roi)
df = pd.DataFrame(data)
analyse_data(df, frame_latency)
