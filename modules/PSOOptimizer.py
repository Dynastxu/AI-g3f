import numpy as np
from rich.progress import Progress


class PSOOptimizer:
    """粒子群优化器"""

    def __init__(self, n_particles=10, n_iterations=20, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles  # 粒子数量
        self.n_iterations = n_iterations  # 迭代次数
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子

    def optimize(self, objective_function, bounds):
        """
        粒子群优化主函数
        bounds: 参数边界，格式为 [(min1, max1), (min2, max2), ...]
        """
        n_dim = len(bounds)

        # 初始化粒子位置和速度
        particles_pos = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (self.n_particles, n_dim)
        )
        particles_vel = np.random.uniform(-1, 1, (self.n_particles, n_dim))

        # 初始化个体最佳位置和适应度
        personal_best_pos = particles_pos.copy()
        personal_best_fitness = np.array([float('inf')] * self.n_particles)

        # 初始化全局最佳
        global_best_pos = None
        global_best_fitness = float('inf')

        # 记录优化过程
        convergence_curve = []

        with Progress() as progress:
            task = progress.add_task("进行粒子群优化", total=self.n_iterations*self.n_particles)
            for iteration in range(self.n_iterations):
                for i in range(self.n_particles):
                    # 计算适应度
                    fitness = objective_function(particles_pos[i])

                    # 更新个体最佳
                    if fitness < personal_best_fitness[i]:
                        personal_best_fitness[i] = fitness
                        personal_best_pos[i] = particles_pos[i].copy()

                    # 更新全局最佳
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_pos = particles_pos[i].copy()

                    progress.update(task, advance=1)

                # 更新粒子速度和位置
                for i in range(self.n_particles):
                    r1, r2 = np.random.random(), np.random.random()

                    # 速度更新
                    cognitive_velocity = self.c1 * r1 * (personal_best_pos[i] - particles_pos[i])
                    social_velocity = self.c2 * r2 * (global_best_pos - particles_pos[i])
                    particles_vel[i] = self.w * particles_vel[i] + cognitive_velocity + social_velocity

                    # 位置更新
                    particles_pos[i] = particles_pos[i] + particles_vel[i]

                    # 边界处理
                    for dim in range(n_dim):
                        if particles_pos[i, dim] < bounds[dim][0]:
                            particles_pos[i, dim] = bounds[dim][0]
                        elif particles_pos[i, dim] > bounds[dim][1]:
                            particles_pos[i, dim] = bounds[dim][1]

                convergence_curve.append(global_best_fitness)

                if (iteration + 1) % 5 == 0:
                    print(f"迭代 {iteration + 1}/{self.n_iterations}, 最佳适应度: {global_best_fitness:.6f}")

        return global_best_pos, global_best_fitness, convergence_curve
