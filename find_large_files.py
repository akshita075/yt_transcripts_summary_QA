import os
import subprocess

def get_large_files(threshold=100):
    # Run git rev-list command to get all objects in the repo
    result = subprocess.run(["git", "rev-list", "--objects", "--all"], stdout=subprocess.PIPE, text=True)
    objects = result.stdout.splitlines()

    large_files = []

    for line in objects:
        parts = line.split()
        if len(parts) < 2:
            continue
        file_hash, file_path = parts[0], " ".join(parts[1:])
        
        # Get file size
        size_result = subprocess.run(["git", "cat-file", "-s", file_hash], stdout=subprocess.PIPE, text=True)
        file_size = int(size_result.stdout.strip())

        if file_size > threshold * 1024 * 1024:  # Convert MB to bytes
            large_files.append((file_size, file_path))

    large_files.sort(reverse=True, key=lambda x: x[0])

    for size, path in large_files[:10]:
        print(f"{size / (1024*1024):.2f} MB - {path}")

if __name__ == "__main__":
    get_large_files()
