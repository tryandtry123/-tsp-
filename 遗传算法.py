import numpy as np
import matplotlib.pyplot as plt

# 目标函数
def objective_function(x, y):
    return ((6.452 * (x + 0.125 * y) * (np.cos(x) - np.cos(2 * y)) ** 2) / np.sqrt(
        (0.8 + (x - 4.2) ** 2 + 2 * (y - 7)) ** 2)) + 3.226 * y

# 适应度函数
def fitness_function(x, y):
    return objective_function(x, y)

# 遗传算法框架
def genetic_algorithm(population_size, generations, crossover_rate, mutation_rate, search_range):
    # 初始化种群
    population = np.random.uniform(low=search_range[0], high=search_range[1], size=(population_size, 2))

    best_fitness_history = []
    best_individual_history = []

    for generation in range(generations):
        # 计算适应度
        fitness_values = np.array([fitness_function(x, y) for x, y in population])

        # Check for NaN values and handle them
        if np.isnan(fitness_values).any() or np.ptp(fitness_values) == 0:
            print(f"Warning: Invalid fitness values encountered in generation {generation}.")
            break

        # 选择操作：使用适应度函数正规化版本作为选择概率
        normalized_fitness = (fitness_values - np.min(fitness_values)) / (
                    np.max(fitness_values) - np.min(fitness_values))

        # Check for NaN values after normalization
        if np.isnan(normalized_fitness).any():
            print(f"Warning: NaN values encountered in normalized fitness in generation {generation}.")
            break

        # Continue with the selection operation
        selection_probabilities = normalized_fitness / np.sum(normalized_fitness)

        # 修正选择操作
        selected_indices = np.random.choice(np.arange(len(population)), size=population_size, replace=True,
                                            p=selection_probabilities)
        selected_population = population[selected_indices]

        # 交叉操作：单点交叉
        crossover_indices = np.random.choice(population_size, size=population_size // 2, replace=False)
        crossover_pairs = selected_population[crossover_indices]
        crossover_points = np.random.rand(population_size // 2, 1)

        # 修正交叉操作
        crossover_offspring = np.zeros_like(crossover_pairs)
        for i in range(crossover_pairs.shape[0]):
            crossover_offspring[i] = crossover_pairs[i, 0] * (1 - crossover_points[i]) + crossover_pairs[i, 1] * \
                                     crossover_points[i]

        # 变异操作：均匀变异
        mutation_mask = np.random.rand(population_size, 2) < mutation_rate
        mutation_offspring = selected_population + mutation_mask * np.random.uniform(low=-0.5, high=0.5,
                                                                                     size=(population_size, 2))

        # 合并新一代种群
        population = np.concatenate([crossover_offspring, mutation_offspring], axis=0)

        # 保留最优个体
        best_index = np.argmax(fitness_values)
        best_fitness = fitness_values[best_index]
        best_individual = population[best_index]

        best_fitness_history.append(best_fitness)
        best_individual_history.append(best_individual)

    return best_fitness_history, best_individual_history

# (2) 最佳适应度和最佳个体图
# 请插入代码以生成适应度和个体的图形

# (3) 不同种群规模的运行结果
population_sizes = [5, 20, 100]
table2_data = []

for population_size in population_sizes:
    best_fitness_history, best_individual_history = genetic_algorithm(population_size, generations=100,
                                                                      crossover_rate=0.8, mutation_rate=0.01,
                                                                      search_range=[0, 10])

    # 计算平均适应度
    average_fitness = np.mean([fitness_function(x, y) for x, y in best_individual_history])

    # 保存结果
    table2_data.append((population_size, best_fitness_history[-1], average_fitness, best_individual_history[-1]))

# # 打印表2
# print("表2 不同的种群规模的GA运行结果")
# print("种群规模\t最佳适应度\t平均适应度\t最佳个体")
# for row in table2_data:
#     print("\t".join(map(str, row)))

# (4) 不同选择策略、交叉策略和变异策略的运行结果
selection_strategies = ['个体选择概率分配', '排序', '比率']
crossover_strategies = ['单点交叉', '两点交叉']
mutation_strategies = ['均匀变异', '高斯变异']

table3_data = []

for s_index, selection_strategy in enumerate(selection_strategies):
    for c_index, crossover_strategy in enumerate(crossover_strategies):
        for m_index, mutation_strategy in enumerate(mutation_strategies):
            # 运行算法10次，取平均值
            avg_best_fitness = 0
            avg_worst_fitness = 0
            avg_average_fitness = 0

            for _ in range(10):
                best_fitness_history, _ = genetic_algorithm(population_size=20, generations=100,
                                                            crossover_rate=0.8, mutation_rate=0.01,
                                                            search_range=[0, 10])

                avg_best_fitness += best_fitness_history[-1]
                avg_worst_fitness += np.min(best_fitness_history)
                avg_average_fitness += np.mean(best_fitness_history)

            avg_best_fitness /= 10
            avg_worst_fitness /= 10
            avg_average_fitness /= 10

            # 保存结果
            table3_data.append((s_index + 1, c_index + 1, m_index + 1,
                                selection_strategy, crossover_strategy, mutation_strategy,
                                avg_best_fitness, avg_worst_fitness, avg_average_fitness))

    # 打印表3
print("\n表3 不同的选择策略、交叉策略和变异策略的算法运行结果")
print("遗传算法参数设置\t1\t2\t3\t4")
print("选择操作\t个体选择概率分配\t排序\t\t\t\t")
print("\t\t比率\t\t\t")
print("个体选择\t轮盘赌选择\t\t\t\t")
print("\t\t竞标赛选择\t\t\t")
print("交叉操作\t单点交叉\t\t\t\t")
print("\t\t两点交叉\t\t\t")
print("变异操作\t均匀变异\t\t\t")
print("\t\t高斯变异\t\t\t")
print("最好适应度\t\t\t\t\t\t", end="")
for i in range(4):
    print(f"{table3_data[i][-3]:.2f}\t", end="")
print("\n最差适应度\t\t\t\t\t\t", end="")
for i in range(4):
    print(f"{table3_data[i][-2]:.2f}\t", end="")
print("\n平均适应度\t\t\t\t\t\t", end="")
for i in range(4):
    print(f"{table3_data[i][-1]:.2f}\t", end="")
print("\n")

