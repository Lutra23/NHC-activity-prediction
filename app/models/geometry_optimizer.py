import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from app.config import XTB_PATH

logger = logging.getLogger(__name__)

class GeometryOptimizer:
    """分子几何构型优化器"""
    
    def __init__(self, xtb_path: str = None):
        """初始化优化器
        
        Args:
            xtb_path: xtb可执行文件的路径
        """
        self.xtb_path = str(XTB_PATH) if xtb_path is None else xtb_path
        self._check_xtb()
        
    def _check_xtb(self):
        """检查xtb是否可用"""
        try:
            print(f"\n检查XTB可执行文件:")
            print(f"XTB路径: {self.xtb_path}")
            
            if not os.path.exists(self.xtb_path):
                raise RuntimeError(f"XTB可执行文件不存在: {self.xtb_path}")
                
            print(f"XTB文件存在: {os.path.exists(self.xtb_path)}")
            print(f"XTB文件大小: {os.path.getsize(self.xtb_path)} 字节")
            
            # 首先尝试使用GBK编码
            print("\n尝试执行XTB命令...")
            result = subprocess.run([self.xtb_path, "--version"], 
                                  capture_output=True, text=True,
                                  encoding='gbk',
                                  errors='ignore',  # 忽略无法解码的字符
                                  shell=True)  # Windows需要shell=True
            
            print(f"命令返回码: {result.returncode}")
            print(f"标准输出: {result.stdout}")
            print(f"错误输出: {result.stderr}")
            
            if result.returncode != 0:
                # 如果命令执行失败，尝试不同的编码
                print("\n尝试其他编码...")
                encodings = ['utf-8', 'cp936', 'gb18030']
                success = False
                
                for encoding in encodings:
                    try:
                        print(f"\n尝试使用编码: {encoding}")
                        result = subprocess.run([self.xtb_path, "--version"], 
                                              capture_output=True, text=True,
                                              encoding=encoding,
                                              errors='ignore',
                                              shell=True)
                        print(f"返回码: {result.returncode}")
                        print(f"输出: {result.stdout}")
                        if result.returncode == 0:
                            success = True
                            break
                    except Exception as e:
                        print(f"使用编码 {encoding} 失败: {str(e)}")
                        continue
                
                if not success:
                    raise RuntimeError("XTB 执行失败")
            
            # 清理输出中的特殊字符
            version_info = result.stdout.strip()
            version_info = ''.join(c for c in version_info if c.isprintable())
            logger.info(f"XTB 版本: {version_info}")
            print(f"XTB 版本信息: {version_info}")
            
        except Exception as e:
            error_msg = f"XTB 检查失败: {str(e)}"
            logger.error(error_msg)
            print(f"\n错误详情: {error_msg}")
            if isinstance(e, subprocess.CalledProcessError):
                print(f"命令返回码: {e.returncode}")
                print(f"命令输出: {e.output}")
            raise RuntimeError(f"XTB 不可用: {str(e)}")
    
    def _write_xyz(self, mol: Chem.Mol, filepath: str):
        """将分子写入XYZ文件"""
        conf = mol.GetConformer()
        with open(filepath, 'w') as f:
            f.write(f"{mol.GetNumAtoms()}\n\n")
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    
    def _read_xyz(self, filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """读取XYZ文件"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            num_atoms = int(lines[0])
            symbols = []
            coords = []
            
            for line in lines[2:2+num_atoms]:
                parts = line.split()
                symbols.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
                
            return np.array(symbols), np.array(coords)
            
        except Exception as e:
            logger.error(f"读取XYZ文件失败: {str(e)}")
            return None
    
    def optimize_geometry(self, mol: Chem.Mol, charge: int = 1, uhf: int = 1, method: str = "GFN2-xTB", solvent: str = "thf") -> Optional[Chem.Mol]:
        """优化分子几何构型
        
        Args:
            mol: RDKit分子对象
            charge: 分子电荷，默认为1
            uhf: 未配对电子数，默认为1
            method: 优化方法，可选值：
                   - "GFN2-xTB": 第二代GFN-xTB方法（默认，推荐）
                   - "GFN1-xTB": 第一代GFN-xTB方法
                   - "GFN0-xTB": 第零代GFN-xTB方法
                   - "GFN-FF": GFN力场方法（最快但最不准确）
                   - "IPEA-xTB": 专门用于激发态的xTB方法
            solvent: 溶剂，默认为THF
        """
        try:
            # 验证方法
            valid_methods = ["GFN2-xTB", "GFN1-xTB", "GFN0-xTB", "GFN-FF", "IPEA-xTB"]
            if method not in valid_methods:
                raise ValueError(f"不支持的方法: {method}. 可用方法: {', '.join(valid_methods)}")
            
            # 根据方法调整参数
            if method == "GFN-FF":
                # 力场方法使用更宽松的参数
                opt_level = "crude"
                max_cycles = 5000
                accuracy = 1.0
                micro_cycles = 100
            elif method == "IPEA-xTB":
                # IPEA方法专门用于激发态
                opt_level = "tight"
                max_cycles = 2000
                accuracy = 0.1
                micro_cycles = 200
            else:
                # GFN0/1/2方法使用标准参数
                opt_level = "tight"
                max_cycles = 2000
                accuracy = 0.1
                micro_cycles = 200
            
            # 生成初始3D构象并进行预优化
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=3000)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                input_xyz = Path(tmpdir) / "input.xyz"
                output_xyz = Path(tmpdir) / "xtbopt.xyz"
                
                # 写入输入文件
                self._write_xyz(mol, input_xyz)
                print(f"\n初始XYZ文件内容:")
                with open(input_xyz, 'r', encoding='utf-8') as f:
                    print(f.read())
                
                # 创建XTB输入文件
                xtb_input = Path(tmpdir) / "xtb.inp"
                with open(xtb_input, "w", encoding='utf-8') as f:
                    f.write("$opt\n")
                    f.write(f"   maxcycle={max_cycles}\n")
                    f.write(f"   microcycle={micro_cycles}\n")
                    f.write(f"   optlevel={opt_level}\n")
                    f.write("   hlow=0.01\n")
                    f.write("   maxdispl=0.08\n")
                    f.write("   maxgrad=0.0001\n")
                    f.write("   s6=20\n")
                    f.write("   modeg=1\n")
                    f.write("$end\n")
                    f.write("$chrg\n")
                    f.write(f"   charge={charge}\n")
                    f.write("$end\n")
                    f.write("$spin\n")
                    f.write(f"   uhf={uhf}\n")
                    f.write("$end\n")
                    f.write("$scf\n")
                    f.write("   maxiterations=1000\n")
                    f.write("   threall=1.0e-12\n")
                    f.write("   denconv=1.0e-8\n")
                    f.write("   broydamp=0.4\n")
                    f.write("   econv=1.0e-7\n")
                    f.write("$end\n")
                    f.write("$solvent\n")
                    f.write(f"   solvent={solvent}\n")
                    f.write("   temperature=298.15\n")
                    f.write("   reference=1.0\n")
                    f.write("   state=1.0\n")
                    f.write("$end\n")
                    f.write("$electronic\n")
                    f.write("   etemp=300.0\n")
                    f.write("   broydamp=0.4\n")
                    f.write("   accuracy=1.0e-4\n")
                    f.write("   maxiter=250\n")
                    f.write("$end\n")
                
                print(f"\nXTB输入文件内容:")
                with open(xtb_input, 'r', encoding='utf-8') as f:
                    print(f.read())
                
                # 准备XTB命令
                method_flag = method.lower().replace("-xtb", "")
                if method == "GFN-FF":
                    method_flag = "ff"
                
                cmd = [
                    self.xtb_path,
                    str(input_xyz),
                    "--opt", opt_level,
                    f"--{method_flag}",
                    f"--chrg", f"{charge}",
                    f"--uhf", f"{uhf}",
                    "--input", str(xtb_input),
                    "--cycles", str(max_cycles),
                    "--parallel", "4",
                    "--acc", str(accuracy),
                    "--etemp", "300.0",
                    "--alpb", solvent,
                    "--verbose"
                ]
                
                print(f"\n运行XTB命令: {' '.join(cmd)}")
                print(f"使用方法: {method}")
                
                # 设置环境变量
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["OMP_NUM_THREADS"] = "4"
                env["MKL_NUM_THREADS"] = "4"
                env["OMP_STACKSIZE"] = "4G"
                env["XTBPATH"] = str(Path(self.xtb_path).parent.parent)
                
                result = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    encoding='gbk',
                    errors='ignore',
                    env=env,
                    shell=True
                )
                
                # 保存XTB的输出，使用二进制模式写入
                try:
                    with open(Path(tmpdir) / "xtb.out", "wb") as f:
                        f.write(result.stdout.encode('utf-8', errors='ignore'))
                    with open(Path(tmpdir) / "xtb.err", "wb") as f:
                        f.write(result.stderr.encode('utf-8', errors='ignore'))
                except Exception as e:
                    print(f"保存XTB输出时出错: {str(e)}")
                
                if result.returncode != 0:
                    try:
                        print("\nXTB标准输出:")
                        print(result.stdout)
                        print("\nXTB错误输出:")
                        print(result.stderr)
                    except Exception as e:
                        print(f"打印XTB输出时出错: {str(e)}")
                    logger.error(f"XTB优化失败: {result.stderr}")
                    return None
                
                # 读取优化结果
                if not output_xyz.exists():
                    logger.error("未找到优化后的结构文件")
                    return None
                    
                # 验证原子数
                try:
                    with open(output_xyz, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        try:
                            xyz_num_atoms = int(first_line)
                            if xyz_num_atoms != mol.GetNumAtoms():
                                logger.error(f"优化前后原子数不匹配: 优化前 {mol.GetNumAtoms()}, 优化后 {xyz_num_atoms}")
                                return None
                        except ValueError:
                            logger.error("无法读取XYZ文件中的原子数")
                            return None
                except Exception as e:
                    print(f"读取XYZ文件时出错: {str(e)}")
                    return None
                
                symbols, coords = self._read_xyz(output_xyz)
                if symbols is None:
                    return None
                
                # 更新分子构象
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, coords[i])
                
                return mol
                
        except Exception as e:
            logger.error(f"几何优化失败: {str(e)}")
            print(f"详细错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def optimize_and_save(self, mol: Chem.Mol, hash_value: str, 
                         output_dir: str = "xyz-opt") -> bool:
        """优化分子构型并保存到XYZ文件
        
        Args:
            mol: RDKit分子对象
            hash_value: 分子哈希值
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        try:
            # 创建输出目录
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 优化几何构型
            opt_mol = self.optimize_geometry(mol)
            if opt_mol is None:
                return False
            
            # 保存优化后的结构
            output_file = output_dir / f"{hash_value}.xyz"
            self._write_xyz(opt_mol, output_file)
            
            return True
            
        except Exception as e:
            logger.error(f"优化并保存失败: {str(e)}")
            return False 
    
    def optimize_if_needed(self, mol, hash_value, xyz_dir="xyz-carbene", charge=1, uhf=0):
        """如果XYZ文件不存在，则进行优化"""
        try:
            xyz_file = Path(xyz_dir) / f"{hash_value}.xyz"
            if not xyz_file.exists():
                print(f"XYZ文件不存在，进行几何优化: {hash_value}")
                if self.optimize_and_save(mol, hash_value, xyz_dir, charge=charge, uhf=uhf):
                    return self.generate_3d_mol(mol, hash_value, xyz_dir)
            return self.generate_3d_mol(mol, hash_value, xyz_dir)
        except Exception as e:
            print(f"优化失败: {str(e)}")
            return None
            
    def generate_3d_mol(self, mol, hash_value, xyz_dir="xyz-carbene"):
        """从XYZ文件生成3D分子结构"""
        try:
            xyz_file = Path(xyz_dir) / f"{hash_value}.xyz"
            if not xyz_file.exists():
                return None
                
            mol = Chem.AddHs(mol)
            xyz_data = xyz_file.read_text().splitlines()
            xyz_num_atoms = int(xyz_data[0].strip())
            mol_num_atoms = mol.GetNumAtoms()
            
            if xyz_num_atoms != mol_num_atoms:
                print(
                    f"原子数不匹配 - Hash: {hash_value}, "
                    f"SMILES: {mol_num_atoms}, XYZ: {xyz_num_atoms}"
                )
                return None
                
            conf = Chem.Conformer(mol_num_atoms)
            for i, line in enumerate(xyz_data[2:2+mol_num_atoms]):
                x, y, z = map(float, line.split()[1:4])
                conf.SetAtomPosition(i, (x, y, z))
                
            mol.AddConformer(conf)
            return mol
            
        except Exception as e:
            print(f"从XYZ文件生成3D结构失败: {str(e)}")
            return None 