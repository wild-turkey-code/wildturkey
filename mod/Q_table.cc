#include "Q_table.h"
#include <algorithm>
#include <random>
#include <cmath>
template <typename T>
inline const T& clamp(const T& val, const T& low, const T& high)
{
    if (val < low) {
        return low;
    }
    if (val > high) {
        return high;
    }
    return val;
}

namespace adgMod {

    QTableManager::QTableManager() {
    }

    double QTableManager::getLearningRate(int state, double action)
    {
        int    n         = Q_table[state].visit_counts[action];
        double alpha0    = 0.3;        // 初始步长（可 0.2~0.5 之间微调）
        double alpha_min = 0.02;       // 新下限，后期更稳定

        // Robbins-Monro 建议：α_k = α0 / sqrt(k)
        double alpha = alpha0 / std::sqrt(n + 1.0);

        return std::max(alpha, alpha_min);
    }

    double QTableManager::compute_reward(
        int      state,           // discrete state s
        double   action,          // chosen error‐bound δ
        double   load_cost,       // observed model load time
        int      layer_count,     // learned-index layer count (>=1)
        uint64_t sstable_size,    // number of keys in this SSTable
        int      level            // LSM-Tree level (L0 = 0 … Lmax)
    )
    {
        //----------------------------------------
        // 0)  常量配置
        //----------------------------------------
        constexpr double w1  = 0.20;          // weight: load_cost
        constexpr double w2  = 0.30;          // weight: layer_count
        constexpr double w3  = 0.50;          // weight: error_bound
        constexpr double wp  = 0.65;          // weight: size penalty
        constexpr double wl  = 0.25;          // weight: level penalty
        constexpr int    MAX_LSM_LEVEL = 3;   // deepest level that matters (L0~L3)

        //----------------------------------------
        // 1) update historical max_sst_size
        //----------------------------------------
        auto &entry       = Q_table[state];
        uint64_t &max_sst = entry.max_sst_size;
        max_sst = std::max(max_sst, sstable_size);

        //----------------------------------------
        // 2) lookup & init historical extremes
        //----------------------------------------
        double &min_load = entry.min_load_model_cost[action];
        double &max_load = entry.max_load_model_cost[action];
        int    &min_layer= entry.min_layer_cost[action];
        int    &max_layer= entry.max_layer_cost[action];
        double &min_err  = entry.min_error_bound[action];
        double &max_err  = entry.max_error_bound[action];

        if (min_load  == std::numeric_limits<double>::max()) min_load  = max_load  = load_cost;
        if (min_layer == std::numeric_limits<int>::max())    min_layer = max_layer = layer_count;
        if (min_err   == std::numeric_limits<double>::max()) min_err   = max_err   = action;

        //----------------------------------------
        // 3) update min / max
        //----------------------------------------
        min_load  = std::min(min_load,  load_cost);
        max_load  = std::max(max_load,  load_cost);
        min_layer = std::min(min_layer, layer_count);
        max_layer = std::max(max_layer, layer_count);
        min_err   = std::min(min_err,   action);
        max_err   = std::max(max_err,   action);

        //----------------------------------------
        // 4) normalize metrics to [0,1]
        //----------------------------------------
        double load_range  = std::max(1e-12, max_load  - min_load);
        double layer_range = std::max(1e-12, double(max_layer - min_layer));
        double err_range   = std::max(1e-12, max_err   - min_err);

        double norm_load   = (load_cost   - min_load)   / load_range;
        double norm_layer  = (layer_count - min_layer)  / layer_range;
        double norm_error  = (action      - min_err)    / err_range;

        //----------------------------------------
        // 5) base utility  U = 1 - Σ w·norm
        //----------------------------------------
        double cost_sum = w1 * norm_load
                        + w2 * norm_layer
                        + w3 * norm_error;
        double u_base   = 1.0 - cost_sum;               // ∈ [−(w1+w2+w3), 1]

        //----------------------------------------
        // 6) size-penalty  inv_size_pen ∈[0,1]
        //----------------------------------------
        double inv_size_pen = std::log(double(max_sst) / double(sstable_size))
                            / std::log(32.0);           // 1/32 ⇒ 1.0
        inv_size_pen = std::pow(inv_size_pen, 1.5);     // extra punch
        inv_size_pen = clamp(inv_size_pen, 0.0, 1.0);

        //----------------------------------------
        // 6-bis) level-penalty  norm_level_pen ∈[0,1]
        //----------------------------------------
        int    lvl_clamped   = std::min(level, MAX_LSM_LEVEL);
        double norm_level_pen= 1.0 - (double)lvl_clamped / MAX_LSM_LEVEL;
        double level_pen     = wl * norm_level_pen;     // ≤ wl

        //----------------------------------------
        // 7) final reward & clamp
        //----------------------------------------
        double r = u_base
                - wp * inv_size_pen
                - level_pen;
        if (level >= 3) r += 0.08;

        r = clamp(r, -(wp + wl), 1.0);
        return r;
    }




    double QTableManager::get_max_future_q(int next_state) const {
        // 获取Q表中next state的Q[]内最高Q值
        double max_q = -std::numeric_limits<double>::infinity();
        for (const auto& pair : Q_table[next_state].q_values) {
            if (pair.second > max_q) {
                max_q = pair.second;
            }
        }
        return (max_q == -std::numeric_limits<double>::infinity()) ? 0.0 : max_q;
    }

    double QTableManager::get_max_future_q(int next_state, const std::vector<double>& actions) const {
        double max_q = -std::numeric_limits<double>::infinity();
        for (const auto& action : actions) {
            auto it = Q_table[next_state].q_values.find(action);
            if (it != Q_table[next_state].q_values.end()) {
                if (it->second > max_q) {
                    max_q = it->second;
                }
            }
        }
        return (max_q == -std::numeric_limits<double>::infinity()) ? 0.0 : max_q;
    }
    // Q-learning 
    double QTableManager::compute_q_value(int    state,
                                      double action_raw,
                                      double reward,
                                      int    next_state)
    {
        /* ---------- 1) 离散化 action 作为 unordered_map key ---------- */
        const double action = std::round(action_raw * 1e4) / 1e4;   // -- 可按需要改精度

        /* ---------- 2) 自适应学习率 / 折扣 ---------- */
        const double alpha  = getLearningRate(state, action);       // 你已有的函数
        const double gamma  = 0.9;                                  // 折扣，可调

        /* ---------- 3) 取得当前 Q 引用，若不存在自动插入 0 ---------- */
        double& prev_q = Q_table[state].q_values[action];           // 默认 0.0

        /* ---------- 4) 计算 next_state 的 max-Q ---------- */
        auto& next_map = Q_table[next_state].q_values;
        double max_future_q = 0.0;
        if (next_map.empty()) {
            // optimistic initial value（若你不想乐观，可以改成 0.0）
            max_future_q = 1.0;
        } else {
            // 手动找最大值，避免再写 get_max_future_q()
            for (const auto& kv : next_map)
                if (kv.second > max_future_q) max_future_q = kv.second;
        }

        /* ---------- 5) TD 更新（带简单 clip 防爆） ---------- */
        const double td_target = reward + gamma * max_future_q;
        double td_error        = td_target - prev_q;

        const double delta_clip = 2.0;                               // ≤2 建议
        if (td_error >  delta_clip) td_error =  delta_clip;
        if (td_error < -delta_clip) td_error = -delta_clip;

        prev_q += alpha * td_error;
        return prev_q;
    }


    // SARSA -> this algorithm focus on the next action's Q value growth
    double QTableManager::compute_q_value(int state, double action, double reward, int next_state, double next_action) {
        double alpha = getLearningRate(state, action);
        double gamma = 0.8; // 折扣因子

        double& prev_q = Q_table[state].q_values[action];
        double next_q = Q_table[next_state].q_values[next_action];
        double Q_value = (1 - alpha) * prev_q + alpha * (reward + gamma * next_q);
        prev_q = Q_value;
        return Q_value;
    }


    void QTableManager::updateQValue(int state, double action, double Q_value) {
        Q_table[state].q_values[action] = Q_value; // 记录state下，action为error_bound时的当前Q值
        Q_table[state].visit_counts[action] += 1;  // 增加访问计数
    }

    double QTableManager::getErrorBound(int state) const {
        return 8;
    }

    // int QTableManager::getNextState(const std::vector<std::string>& string_keys) const {
    //     if (string_keys.empty()) {
    //         // 根据实际需求处理空的 string_keys，例如返回一个默认状态
    //         return 0;
    //     }

    //     uint64_t min_key = atoll(string_keys.front().c_str());   // SST内最小Key, 为Uint64型
    //     uint64_t max_key = atoll(string_keys.back().c_str());    // SST内最大Key, 为Uint64型
    //     uint64_t size = string_keys.size();                      // SST内Key的数量, 为Uint64型
    //     uint64_t inverse_density = (max_key - min_key) / size;   // SST内key分布的模拟密度

    //     // 预定将inverse_density < 10 为一档， < 30为二档， < 50为3档， >=50为4档
    //     if (inverse_density < 10) return 0;
    //     else if (inverse_density < 30) return 1;
    //     else if (inverse_density < 50) return 2;
    //     else return 3;
    // }

    // 添加经验到回放缓冲区
    void QTableManager::addExperience(int state, double action, double reward) {
        Experience exp = {state, action, reward};
        replay_buffer.push_back(exp);
        if (replay_buffer.size() > max_replay_size) {
            replay_buffer.erase(replay_buffer.begin());
        }
    }


    // 采样经验进行学习
    // void QTableManager::learnFromReplay() {
    //     if (replay_buffer.size() < batch_size) return;

    //     // 按照某种优先级排序，例如高奖励优先
    //     std::sort(replay_buffer.begin(), replay_buffer.end(), [&](const Experience& a, const Experience& b) {
    //         return a.reward > b.reward; // 或者根据其他优先级标准
    //     });

    //     for (size_t i = 0; i < batch_size && i < replay_buffer.size(); ++i) {
    //         Experience exp = replay_buffer[i];
    //         double Q_value = compute_q_value(exp.state, exp.action, exp.reward, exp.next_state);
    //         updateQValue(exp.state, exp.action, Q_value);
    //     }
    // }

    void QTableManager::learnFromReplay() {
        if (replay_buffer.size() < batch_size) return;

        // 初始化随机数生成器
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, replay_buffer.size() - 1);

        for (size_t i = 0; i < batch_size; ++i) {
            int idx = dis(gen);
            Experience& exp = replay_buffer[idx];
            double Q_value;
            //Q_value = compute_q_value(exp.state, exp.action, exp.reward, exp.next_state, exp.next_action);
            Q_value = compute_q_value(exp.state, exp.action, exp.reward, exp.next_state);

            updateQValue(exp.state, exp.action, Q_value);
        }
    }

    void QTableManager::initQTable() {
        Q_table.resize(8);
        std::vector<int> actions = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48}; // 可选的 error_bound 值
        //std::vector<double> actions = {8, 12, 16, 20, 24, 32, 36};
        for (auto& entry : Q_table) {
            // 初始化 min 和 max 值的映射
            for (double action : actions) {
                entry.q_values[action] = 0.0;
                entry.min_load_model_cost[action] = std::numeric_limits<double>::max();
                entry.max_load_model_cost[action] = std::numeric_limits<double>::lowest();
                entry.min_layer_cost[action] = std::numeric_limits<double>::max();
                entry.max_layer_cost[action] = std::numeric_limits<double>::lowest();
                entry.min_error_bound[action] = std::numeric_limits<double>::max();
                entry.max_error_bound[action] = std::numeric_limits<double>::lowest();
                entry.visit_counts[action] = 0;
                entry.last_action = 16.0;
            }
        }
    }

    QTableManager& getQTableManagerInstance() {
        static QTableManager instance;
        return instance;
    }

}