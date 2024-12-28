// 处理API错误响应
function handleApiError(error) {
    console.error('API错误:', error);
    return {
        error: error.message || '未知错误'
    };
}

// 格式化预测结果
function formatPredictionResult(result) {
    if (!result || !result.bond_orders) {
        throw new Error('无效的预测结果');
    }
    
    return Object.entries(result.bond_orders).map(([bond, order]) => ({
        bond: bond,
        order: parseFloat(order).toFixed(3)
    }));
}

// 显示错误消息
function showError(message, containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger">
                ${message}
            </div>
        `;
    }
}

// 显示成功消息
function showSuccess(message, containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-success">
                ${message}
            </div>
        `;
    }
}

// 显示预测结果
function displayPredictionResults(results, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    try {
        const formattedResults = formatPredictionResult(results);
        
        container.innerHTML = `
            <h6>预测结果:</h6>
            <table class="table table-sm">
                <thead>
                    <tr>
                        <th>化学键</th>
                        <th>键级</th>
                    </tr>
                </thead>
                <tbody>
                    ${formattedResults.map(r => `
                        <tr>
                            <td>${r.bond}</td>
                            <td>${r.order}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (error) {
        showError(error.message, containerId);
    }
}

// 预测单个分子的键级
async function predictBondOrder(smiles, carbeneIdx) {
    try {
        const response = await fetch('/api/v1/predict/carbene-bond', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                smiles: smiles,
                carbene_idx: parseInt(carbeneIdx)
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '预测失败');
        }
        
        return data;
    } catch (error) {
        throw new Error(`预测失败: ${error.message}`);
    }
}

// 批量预测分子的键级
async function predictBondOrdersBatch(molecules, nJobs = -1) {
    try {
        const response = await fetch('/api/v1/predict/carbene-bond/batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                molecules: molecules,
                n_jobs: nJobs
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '批量预测失败');
        }
        
        return data;
    } catch (error) {
        throw new Error(`批量预测失败: ${error.message}`);
    }
} 