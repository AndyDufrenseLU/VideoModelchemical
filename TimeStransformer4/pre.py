import os

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def process_folder(folder, out_file, particle_label):
    """
    遍历 folder 下各个以数字命名的子文件夹，写出：
    绝对路径  粒子类别  气速
    """
    if not os.path.isdir(folder):
        print(f"目录不存在：{folder}")
        return 0

    n = 0
    for subdir in sorted(os.listdir(folder)):
        subfolder = os.path.join(folder, subdir)
        if not os.path.isdir(subfolder):
            continue
        try:
            gas_speed = int(subdir)  # 子文件夹名即气速
        except ValueError:
            print(f"子文件夹名不是数字：{subdir}，跳过")
            continue

        for fname in sorted(os.listdir(subfolder)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            abs_path = os.path.abspath(os.path.join(subfolder, fname))
            out_file.write(f"{abs_path} {particle_label} {gas_speed}\n")
            n += 1
    return n


def main():
    # 原有粒子
    narrow_folder = r"C:\Bowen running\Data\fccanarrow_train"   # 粒子 0
    wide_folder   = r"C:\Bowen running\Data\fccawide_train"     # 粒子 1
    c_folder      = r"C:\Bowen running\Data\GB28narrow_train"        # 粒子 2

    # 新增粒子 3（label = 3）
    particle3_folder = r"C:\Bowen running\Data\GB28wide_train" # ← 修改为你的真实路径

    out_txt = r"C:\Bowen running\TimeStransformer\alltrain_with_c.txt"

    total = 0
    with open(out_txt, 'w', encoding='utf-8') as fout:
        total += process_folder(narrow_folder,   fout, particle_label=0)
        total += process_folder(wide_folder,     fout, particle_label=1)
        total += process_folder(c_folder,        fout, particle_label=2)
        total += process_folder(particle3_folder,fout, particle_label=3)  # 新增粒子 3

    print(f"合并输出完成：{out_txt}（共写入 {total} 行）")


if __name__ == '__main__':
    main()
