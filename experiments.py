from hashing import main
import os

target_hash = 1
true_hash = 2
while target_hash != true_hash:
    start_file = open('start','wb')
    start_file.write(os.urandom(300))
    start_file.close()
    start_hash,target_hash = main(['./test_files/target_random', '--source', 'start', '--output', 'out', '--iters', '1000'])
    true_hash,_ = main(['out'])
    print(target_hash)
    print(true_hash)
