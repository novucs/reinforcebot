import os

import matplotlib.pyplot as plt

episode_count = 250

simple_runs = 0
simple_data = [0] * episode_count

pixel_runs = 0
pixel_data = [0] * episode_count

for filename in os.listdir('../logs'):
    simple = filename.startswith('simple_')
    pixel = filename.startswith('pixel_')
    filename = '../logs/' + filename

    if not (simple or pixel):
        continue

    with open(filename, 'r') as file:
        lines = file.readlines()

        if len(lines) < episode_count:
            continue

        if simple:
            simple_runs += 1
        elif pixel:
            pixel_runs += 1

        for line in lines:
            if ',' not in line:
                continue

            episode, steps = line.split(',')
            episode, steps = int(episode), int(steps)

            if episode >= episode_count:
                continue

            if simple:
                simple_data[episode] += steps
            else:
                pixel_data[episode] += steps

print(simple_runs, pixel_runs)
simple_data = [x / simple_runs for x in simple_data]
pixel_data = [x / pixel_runs for x in pixel_data]

# plt.plot(range(episode_count), simple_data)
# plt.plot(range(episode_count), pixel_data)

plt.show()
fig = plt.figure()
plt.plot(range(episode_count), simple_data, label='reduced observations')
plt.plot(range(episode_count), pixel_data, label='visual observations')
plt.legend(loc='upper left')
fig.suptitle('ReinforceBot learning CartPole', fontsize=20)
plt.xlabel('episode', fontsize=18)
plt.ylabel('mean step count', fontsize=16)
fig.savefig('test.jpg')
