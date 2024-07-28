import numpy as np



def bezier_5(contour,k,inserted,closed):
        # 0 前处理==========================================================================
        date_x = np.array([point[0][0] for point in contour])
        date_y = np.array([point[0][1] for point in contour])

        # 第1步：生成原始数据折线中点集==========================================================================
        mid_points = []

        def add_point(date_x, date_y, point_list, i):
            point_list.append({
                'start': (date_x[i - 1], date_y[i - 1]),
                'end': (date_x[i], date_y[i]),
                'mid': ((date_x[i] + date_x[i - 1]) / 2.0, (date_y[i] + date_y[i - 1]) / 2.0)
            })
            return point_list

        # 通过循环处理大部分点对
        for i in range(1, len(date_x)):
            mid_points = add_point(date_x, date_y, mid_points, i)
        # 如果闭合，特别处理最后一个点到第一个点的情况
        if closed:
            mid_points = add_point(date_x, date_y, mid_points, 0)

        # 第2步：找出中点连线及其分割点 ===============================================================================
        split_points = []
        n = len(mid_points)  # 获取中点列表的长度
        for i in range(n):
            if i < n - 1:
                next_point = mid_points[i + 1]
            elif closed:
                next_point = mid_points[0]
            else:
                break  # 如果非闭合且是最后一个点，退出循环
            current = mid_points[i]
            # 计算段长度
            d0 = np.sqrt((current['start'][0] - current['end'][0]) ** 2 +
                         (current['start'][1] - current['end'][1]) ** 2)
            d1 = np.sqrt((next_point['start'][0] - next_point['end'][0]) ** 2 +
                         (next_point['start'][1] - next_point['end'][1]) ** 2)
            k_split = d0 / (d0 + d1)
            # 计算分割点
            mx0, my0 = current['mid']
            mx1, my1 = next_point['mid']
            split_x = mx0 + (mx1 - mx0) * k_split
            split_y = my0 + (my1 - my0) * k_split
            split_points.append({
                'start': (mx0, my0),
                'end': (mx1, my1),
                'split': (split_x, split_y)
            })

        # 第3步：平移中点连线，调整端点，生成控制点 ===============================================================================
        crt_points = list()
        for i in range(len(split_points)):
            vx, vy = mid_points[i]['end']  # 当前顶点的坐标
            dx = vx - split_points[i]['split'][0]  # 平移线段x偏移量
            dy = vy - split_points[i]['split'][1]  # 平移线段y偏移量

            sx, sy = split_points[i]['start'][0] + dx, split_points[i]['start'][1] + dy  # 平移后线段起点坐标
            ex, ey = split_points[i]['end'][0] + dx, split_points[i]['end'][1] + dy  # 平移后线段终点坐标

            cp0 = sx + (vx - sx) * k, sy + (vy - sy) * k  # 控制点坐标
            cp1 = ex + (vx - ex) * k, ey + (vy - ey) * k  # 控制点坐标

            if crt_points:
                crt_points[-1].insert(2, cp0)
            else:
                crt_points.append([mid_points[0]['start'], cp0, mid_points[0]['end']])

            if closed:
                if i < (len(mid_points) - 1):
                    crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
                else:
                    crt_points[0].insert(1, cp1)
            else:
                if i < (len(mid_points) - 2):
                    crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
                else:
                    crt_points.append(
                        [mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end'], mid_points[i + 1]['end']])
                    crt_points[0].insert(1, mid_points[0]['start'])
        # 第4步：应用贝塞尔曲线方程插值 ===============================================================================
        out = []
        for item in crt_points:
            item = [np.array(i) if isinstance(i, (tuple, list)) else i for i in
                    item]  # 确认isinstance(item[0], (tuple, list))
            points = []
            for t in np.linspace(0, 1, inserted + 2):
                points.append(item[0] * np.power((1 - t), 3) + 3 *
                              item[1] * t * np.power((1 - t), 2) + 3 *
                              item[2] * (1 - t) * np.power(t, 2) +
                              item[3] * np.power(t, 3))
            group = np.vstack(points)
            out.append(group[:-1])
        out.append(group[-1:])
        out = np.vstack(out)
        x_curve, y_curve = out.T[0], out.T[1]

        # 第5步：后处理
        curve_list = [[[int(x), int(y)]] for x, y in zip(x_curve, y_curve)]
        contour = np.array(curve_list)
        return contour
def bezier_1(contour, k, inserted,closed):
    # 0 前处理
    date_x = np.array([point[0][0] for point in contour])
    date_y = np.array([point[0][1] for point in contour])

    # 第1步：生成原始数据折线中点集===============================================================================
    mid_points = []

    def add_point(date_x, date_y, point_list, i):
        point_list.append({
            'start': (date_x[i - 1], date_y[i - 1]),
            'end': (date_x[i], date_y[i]),
            'mid': ((date_x[i] + date_x[i - 1]) / 2.0, (date_y[i] + date_y[i - 1]) / 2.0)
        })
        return point_list

    # 通过循环处理大部分点对
    for i in range(1, len(date_x)):
        mid_points = add_point(date_x, date_y, mid_points, i)

    # 如果闭合，特别处理最后一个点到第一个点的情况
    if closed:
        mid_points = add_point(date_x, date_y, mid_points, 0)

    # 第2步：找出中点连线及其分割点 ===============================================================================
    split_points = []
    n = len(mid_points)  # 获取中点列表的长度

    for i in range(n):
        if i < n - 1:
            next_point = mid_points[i + 1]
        elif closed:
            next_point = mid_points[0]
        else:
            break  # 如果非闭合且是最后一个点，退出循环

        current = mid_points[i]

        # 计算段长度
        d0 = np.sqrt((current['start'][0] - current['end'][0]) ** 2 +
                     (current['start'][1] - current['end'][1]) ** 2)
        d1 = np.sqrt((next_point['start'][0] - next_point['end'][0]) ** 2 +
                     (next_point['start'][1] - next_point['end'][1]) ** 2)

        k_split = d0 / (d0 + d1)

        # 计算分割点
        mx0, my0 = current['mid']
        mx1, my1 = next_point['mid']
        split_x = mx0 + (mx1 - mx0) * k_split
        split_y = my0 + (my1 - my0) * k_split

        split_points.append({
            'start': (mx0, my0),
            'end': (mx1, my1),
            'split': (split_x, split_y)
        })

    # 第3步：平移中点连线，调整端点，生成控制点 ===============================================================================
    # 第3步：平移中点连线，调整端点，生成控制点
    crt_points = list()
    for i in range(len(split_points)):
        vx, vy = mid_points[i]['end']  # 当前顶点的坐标
        dx = vx - split_points[i]['split'][0]  # 平移线段x偏移量
        dy = vy - split_points[i]['split'][1]  # 平移线段y偏移量

        sx, sy = split_points[i]['start'][0] + dx, split_points[i]['start'][1] + dy  # 平移后线段起点坐标
        ex, ey = split_points[i]['end'][0] + dx, split_points[i]['end'][1] + dy  # 平移后线段终点坐标

        cp0 = sx + (vx - sx) * k, sy + (vy - sy) * k  # 控制点坐标
        cp1 = ex + (vx - ex) * k, ey + (vy - ey) * k  # 控制点坐标

        if crt_points:
            crt_points[-1].insert(2, cp0)
        else:
            crt_points.append([mid_points[0]['start'], cp0, mid_points[0]['end']])

        if closed:
            if i < (len(mid_points) - 1):
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
            else:
                crt_points[0].insert(1, cp1)
        else:
            if i < (len(mid_points) - 2):
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
            else:
                crt_points.append(
                    [mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end'], mid_points[i + 1]['end']])
                crt_points[0].insert(1, mid_points[0]['start'])

    # 第4步：应用贝塞尔曲线方程插值 ===============================================================================
    out = []
    for item in crt_points:
        item = [np.array(i) if isinstance(i, (tuple, list)) else i for i in
                item]  # 确认isinstance(item[0], (tuple, list))
        points = []
        for t in np.linspace(0, 1, inserted + 2):
            points.append(item[0] * np.power((1 - t), 3) + 3 *
                          item[1] * t * np.power((1 - t), 2) + 3 *
                          item[2] * (1 - t) * np.power(t, 2) +
                          item[3] * np.power(t, 3))
        group = np.vstack(points)
        out.append(group[:-1])
    out.append(group[-1:])
    out = np.vstack(out)

    x_curve, y_curve = out.T[0], out.T[1]

    # 第5步：后处理
    curve_list = [[[int(x), int(y)]] for x, y in zip(x_curve, y_curve)]
    contour = np.array(curve_list)
    return contour


def bezier_2(contour, k, inserted,closed):
    # 0 前处理
    date_x = np.array([point[0][0] for point in contour])
    date_y = np.array([point[0][1] for point in contour])

    # 第1步：生成原始数据折线中点集===============================================================================
    mid_points = []

    def add_point(date_x, date_y, point_list, i):
        point_list.append({
            'start': (date_x[i - 1], date_y[i - 1]),
            'end': (date_x[i], date_y[i]),
            'mid': ((date_x[i] + date_x[i - 1]) / 2.0, (date_y[i] + date_y[i - 1]) / 2.0)
        })
        return point_list

    # 通过循环处理大部分点对
    for i in range(1, len(date_x)):
        mid_points = add_point(date_x, date_y, mid_points, i)

    # 如果闭合，特别处理最后一个点到第一个点的情况
    if closed:
        mid_points = add_point(date_x, date_y, mid_points, 0)

    # 第2步：找出中点连线及其分割点 ===============================================================================
    split_points = []
    n = len(mid_points)  # 获取中点列表的长度

    for i in range(n):
        if i < n - 1:
            next_point = mid_points[i + 1]
        elif closed:
            next_point = mid_points[0]
        else:
            break  # 如果非闭合且是最后一个点，退出循环

        current = mid_points[i]

        # 计算段长度
        d0 = np.sqrt((current['start'][0] - current['end'][0]) ** 2 +
                     (current['start'][1] - current['end'][1]) ** 2)
        d1 = np.sqrt((next_point['start'][0] - next_point['end'][0]) ** 2 +
                     (next_point['start'][1] - next_point['end'][1]) ** 2)

        k_split = d0 / (d0 + d1)

        # 计算分割点
        mx0, my0 = current['mid']
        mx1, my1 = next_point['mid']
        split_x = mx0 + (mx1 - mx0) * k_split
        split_y = my0 + (my1 - my0) * k_split

        split_points.append({
            'start': (mx0, my0),
            'end': (mx1, my1),
            'split': (split_x, split_y)
        })

    # 第3步：平移中点连线，调整端点，生成控制点 ===============================================================================
    crt_points = []  # 初始化用于存储控制点的列表

    # 遍历所有的中点对
    for i in range(len(mid_points)):
        vx, vy = mid_points[i]['end']  # 提取当前中点的终点坐标
        dx = vx - split_points[i]['split'][0]  # 计算当前中点终点到分割点的X轴偏移量
        dy = vy - split_points[i]['split'][1]  # 计算当前中点终点到分割点的Y轴偏移量

        sx, sy = split_points[i]['start'][0] + dx, split_points[i]['start'][1] + dy  # 根据偏移量计算平移后的起始点坐标
        ex, ey = split_points[i]['end'][0] + dx, split_points[i]['end'][1] + dy  # 根据偏移量计算平移后的终点坐标

        # 计算控制点坐标，通过比例 k 调整起始点和终点之间的距离
        cp0 = (sx + (vx - sx) * k, sy + (vy - sy) * k)
        cp1 = (ex + (vx - ex) * k, ey + (vy - ey) * k)

        # 如果是第一个点，直接添加到控制点列表
        if i == 0:
            crt_points.append([mid_points[i]['start'], cp0, cp1, mid_points[i]['end']])
        else:
            # 更新上一个控制点集的最后两个点，并添加新的控制点集
            crt_points[-1][2:] = [cp0, mid_points[i]['end']]
            crt_points.append([mid_points[i]['start'], cp0, cp1, mid_points[i]['end']])

        # 如果是最后一个点并且形状是闭合的，将第一个控制点集的起点与最后一个控制点集的终点相连
        if i == len(mid_points) - 1 and closed:
            crt_points.append([mid_points[0]['start'], crt_points[0][1], crt_points[0][2], mid_points[0]['end']])

    # 第4步：应用贝塞尔曲线方程插值 ===============================================================================
    out = []
    for item in crt_points:
        item = [np.array(i) if isinstance(i, (tuple, list)) else i for i in
                item]  # 确认isinstance(item[0], (tuple, list))
        points = []
        for t in np.linspace(0, 1, inserted + 2):
            points.append(item[0] * np.power((1 - t), 3) + 3 *
                          item[1] * t * np.power((1 - t), 2) + 3 *
                          item[2] * (1 - t) * np.power(t, 2) +
                          item[3] * np.power(t, 3))
        group = np.vstack(points)
        out.append(group[:-1])
    out.append(group[-1:])
    out = np.vstack(out)

    x_curve, y_curve = out.T[0], out.T[1]

    # 第5步：后处理
    curve_list = [[[int(x), int(y)]] for x, y in zip(x_curve, y_curve)]
    contour = np.array(curve_list)
    return contour


def bezier_3(contour, k, inserted,closed):
    # 0 前处理
    date_x = np.array([point[0][0] for point in contour])
    date_y = np.array([point[0][1] for point in contour])

    n = len(date_x)

    crt_points = []  # 初始化用于存储控制点的列表
    split_points = []  # 初始化分割点列表

    # 单个循环处理所有操作
    for i in range(n):
        # 添加闭合特性
        next_index = 0 if closed and i == n - 1 else i + 1
        if not closed and i == n - 1:
            break

        # 计算中点
        mid_point = ((date_x[i] + date_x[next_index]) / 2, (date_y[i] + date_y[next_index]) / 2)

        if i > 0 or closed:
            # 计算分割点
            prev_mid_x, prev_mid_y = split_points[-1] if split_points else mid_point
            d0 = np.sqrt((date_x[i - 1] - date_x[i]) ** 2 + (date_y[i - 1] - date_y[i]) ** 2)
            d1 = np.sqrt((date_x[next_index] - date_x[i]) ** 2 + (date_y[next_index] - date_y[i]) ** 2)
            k_split = d0 / (d0 + d1)
            split_x, split_y = (prev_mid_x + (mid_point[0] - prev_mid_x) * k_split,
                                prev_mid_y + (mid_point[1] - prev_mid_y) * k_split)
            split_points.append((split_x, split_y))

            # 计算控制点
            vx, vy = date_x[i], date_y[i]
            sx, sy = prev_mid_x, prev_mid_y
            (ex, ey) = mid_point
            dx, dy = vx - split_x, vy - split_y
            cp0, cp1 = ((sx + dx * k, sy + dy * k),
                        (ex + dx * k, ey + dy * k))

            # 存储控制点数据
            if i == 0:
                crt_points.append([split_points[-1], cp0, cp1, mid_point])
            else:
                crt_points[-1][2:] = [cp0, mid_point]
                crt_points.append([split_points[-1], cp0, cp1, mid_point])

    # 第4步：应用贝塞尔曲线方程插值 ===============================================================================
    out = []
    for item in crt_points:
        item = [np.array(i) if isinstance(i, (tuple, list)) else i for i in
                item]  # 确认isinstance(item[0], (tuple, list))
        points = []
        for t in np.linspace(0, 1, inserted + 2):
            points.append(item[0] * np.power((1 - t), 3) + 3 *
                          item[1] * t * np.power((1 - t), 2) + 3 *
                          item[2] * (1 - t) * np.power(t, 2) +
                          item[3] * np.power(t, 3))
        group = np.vstack(points)
        out.append(group[:-1])
    out.append(group[-1:])
    out = np.vstack(out)

    x_curve, y_curve = out.T[0], out.T[1]

    # 第5步：后处理
    curve_list = [[[int(x), int(y)]] for x, y in zip(x_curve, y_curve)]
    contour = np.array(curve_list)
    return contour


def bezier_4(contour, k, inserted,closed):
    points = np.array([[p[0][0], p[0][1]] for p in contour])
    n = len(points)
    out = []

    for i in range(n):
        next_index = (i + 1) % n if i < n - 1 else 0
        # 使用每个点和其后继点的线性组合作为控制点
        p0, p3 = points[i], points[next_index]
        p1 = p0 + k * (points[next_index] - points[i])
        p2 = p3 - k * (points[next_index] - points[i])
        segment = []
        for t in np.linspace(0, 1, inserted + 1):
            bezier_point = (p0 * (1 - t) ** 3 +
                            3 * p1 * t * (1 - t) ** 2 +
                            3 * p2 * t ** 2 * (1 - t) +
                            p3 * t ** 3)
            segment.append(bezier_point)
        out.extend(segment[:-1])
    out.append(segment[-1])

    curve_list = [[[int(x), int(y)]] for x, y in out]
    contour = np.array(curve_list)
    return contour
