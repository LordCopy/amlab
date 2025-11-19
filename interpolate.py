def divided_diff(x, y):
    n = len(y)
    table = [[0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    return table

def newton_interpolation(x, y, value):
    n = len(x)
    table = divided_diff(x, y)
    result = table[0][0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (value - x[i-1])
        result += table[0][i] * product_term
    return result

if __name__ == '__main__':
    x_points = [1,2,3,4]
    y_points = [1,4,9,16]
    print('Interpolated:', newton_interpolation(x_points, y_points, 2.5))
