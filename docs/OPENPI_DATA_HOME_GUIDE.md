# OpenPI Download Cache Configuration Guide

## Overview

This guide explains how to configure the `OPENPI_DATA_HOME` environment variable to control where all OpenPI downloads (checkpoints, datasets, etc.) are stored on your system.

**Key Benefits:**
- ✓ Centralized cache location across your entire system
- ✓ Easy to manage and organize downloads
- ✓ No code changes needed in any project files
- ✓ Works seamlessly with Docker and containerized environments
- ✓ Portable across different PCs with simple configuration

---

## Quick Start

### For Current PC (Already Configured)

Your current setup:
```bash
OPENPI_DATA_HOME=/mnt/22TB_IndEgo/yahuan/openpi/checkpoints
```

This is already set in:
- `~/.bashrc`
- `~/.zshrc` (if you use zsh)

**Verify it's working:**
```bash
echo $OPENPI_DATA_HOME
```

---

## Configuration Instructions

### Method 1: Persistent Configuration (Recommended)

This method makes the setting survive across terminal sessions and restarts.

#### Step 1: Edit Your Shell Profile

**For bash users:**
```bash
nano ~/.bashrc
```

**For zsh users:**
```bash
nano ~/.zshrc
```

#### Step 2: Add the Environment Variable

At the end of the file, add:
```bash
# OpenPI data cache directory
export OPENPI_DATA_HOME=/path/to/your/checkpoints
```

**Examples:**
```bash
# Example 1: Single large storage location
export OPENPI_DATA_HOME=/mnt/large_storage/openpi_cache

# Example 2: User's home directory
export OPENPI_DATA_HOME=~/openpi_data

# Example 3: Absolute path (recommended)
export OPENPI_DATA_HOME=/mnt/22TB_IndEgo/yahuan/openpi/checkpoints
```

#### Step 3: Apply Changes

Option A - Reload the profile:
```bash
source ~/.bashrc
```

Option B - Close and reopen your terminal

#### Step 4: Verify

```bash
echo $OPENPI_DATA_HOME
```

Should output your configured path.

---

### Method 2: Temporary Configuration (Current Session Only)

For one-time use without saving:

```bash
export OPENPI_DATA_HOME=/your/desired/path
```

This only works for the current terminal session and will be lost when you close the terminal.

---

### Method 3: Command-Line Automatic Setup

Run these commands to automatically add it to your profile:

```bash
# For bash
echo "export OPENPI_DATA_HOME=/mnt/22TB_IndEgo/yahuan/openpi/checkpoints" >> ~/.bashrc
source ~/.bashrc

# For zsh
echo "export OPENPI_DATA_HOME=/mnt/22TB_IndEgo/yahuan/openpi/checkpoints" >> ~/.zshrc
source ~/.zshrc
```

---

## Setting Up on Different PCs

### On a New PC or Different User Account

Follow the same steps above, but adjust the path according to that system's structure:

**Step-by-step:**

1. **Identify where you want to store downloads:**
   ```bash
   # Option A: Use a specific directory
   /path/to/storage/openpi_checkpoints
   
   # Option B: Use home directory
   ~/openpi_data
   
   # Option C: Use a mounted drive
   /mnt/external_drive/openpi_cache
   ```

2. **Add to shell profile:**
   ```bash
   echo "export OPENPI_DATA_HOME=/your/chosen/path" >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Create the directory:**
   ```bash
   mkdir -p $OPENPI_DATA_HOME
   ```

4. **Verify:**
   ```bash
   python << 'EOF'
   import sys
   import os
   sys.path.insert(0, '/path/to/openpi/src')
   from openpi.shared import download
   cache_dir = download.get_cache_dir()
   print(f"Cache directory: {cache_dir}")
   print(f"Directory exists: {cache_dir.exists()}")
   EOF
   ```

---

## How It Works

### Download Path Structure

When you call `download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")`, the file is stored at:

```
$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_fast_droid/
```

**Example:**
- `OPENPI_DATA_HOME=/mnt/22TB_IndEgo/yahuan/openpi/checkpoints`
- Download: `gs://openpi-assets/checkpoints/pi0_fast_droid`
- Stored at: `/mnt/22TB_IndEgo/yahuan/openpi/checkpoints/openpi-assets/checkpoints/pi0_fast_droid/`

### Default Behavior

If `OPENPI_DATA_HOME` is not set, the system defaults to:
```bash
~/.cache/openpi
```

---

## Using with Docker

The Docker Compose files in this project already support `OPENPI_DATA_HOME`.

### How It Works

The `compose.yml` files:
1. Mount `$OPENPI_DATA_HOME` from the host
2. Set `OPENPI_DATA_HOME=/openpi_assets` inside the container
3. Downloads in the container write to your host's directory

### Running with Docker

```bash
# Make sure OPENPI_DATA_HOME is set
export OPENPI_DATA_HOME=/mnt/22TB_IndEgo/yahuan/openpi/checkpoints

# Navigate to an example
cd examples/aloha_sim

# Run Docker
docker compose -f compose.yml up --build
```

### Docker Compose Configuration

Example from `examples/aloha_sim/compose.yml`:
```yaml
openpi_server:
  volumes:
    - ${OPENPI_DATA_HOME:-~/.cache/openpi}:/openpi_assets
  environment:
    - OPENPI_DATA_HOME=/openpi_assets
```

This means:
- `${OPENPI_DATA_HOME:-~/.cache/openpi}` - Uses your env var, or defaults to `~/.cache/openpi`
- `:/openpi_assets` - Maps it to `/openpi_assets` inside the container
- Downloads inside the container go to your host system

---

## Python Usage

You don't need to modify any Python code. Just import and use as normal:

```python
from openpi.shared import download

# This will use OPENPI_DATA_HOME automatically
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
```

### Advanced: Direct Cache Directory Access

```python
from openpi.shared import download

# Get the cache directory
cache_dir = download.get_cache_dir()
print(f"Cache location: {cache_dir}")

# List downloaded checkpoints
import os
for item in os.listdir(cache_dir):
    print(item)
```

---

## Troubleshooting

### Issue: `OPENPI_DATA_HOME` Not Being Used

**Solution 1: Verify it's set**
```bash
echo $OPENPI_DATA_HOME
```

If empty, it's not set. Use Method 1 or 3 above to set it.

**Solution 2: Check shell profile was edited correctly**
```bash
cat ~/.bashrc | grep OPENPI_DATA_HOME
```

Should show your configuration line.

**Solution 3: Reload your shell**
```bash
source ~/.bashrc
# or close and reopen terminal
```

### Issue: "Permission denied" when creating cache directory

```bash
# Fix permissions
sudo chmod 755 /path/to/openpi_checkpoints
# Or use your username
sudo chown $USER /path/to/openpi_checkpoints
```

### Issue: Downloads to wrong location

1. Check what's currently set:
   ```bash
   echo $OPENPI_DATA_HOME
   ```

2. Verify in Python:
   ```python
   import sys
   sys.path.insert(0, '/path/to/openpi/src')
   from openpi.shared import download
   print(download.get_cache_dir())
   ```

3. If different from expected, update `~/.bashrc` and `source ~/.bashrc`

### Issue: Docker can't access downloads

Ensure `OPENPI_DATA_HOME` is set in your host shell before running Docker:
```bash
export OPENPI_DATA_HOME=/mnt/22TB_IndEgo/yahuan/openpi/checkpoints
docker compose up
```

---

## Common Scenarios

### Scenario 1: Using Multiple Projects

If you have multiple OpenPI projects on one PC, they all share the same cache:

```bash
# All projects use this configuration
export OPENPI_DATA_HOME=/shared/openpi_cache

# Project 1
cd /home/user/project1
python train.py

# Project 2 (reuses downloads from Project 1)
cd /home/user/project2
python train.py
```

### Scenario 2: Large Storage on External Drive

```bash
# External drive with more space
export OPENPI_DATA_HOME=/mnt/external_drive/openpi_data
```

### Scenario 3: Different Settings Per User

Each user has their own home directory configuration:

```bash
# User A
export OPENPI_DATA_HOME=/home/userA/openpi_cache

# User B
export OPENPI_DATA_HOME=/home/userB/openpi_cache
```

### Scenario 4: Team Shared Cache

On a shared server:
```bash
export OPENPI_DATA_HOME=/shared/team/openpi_cache
# All team members download once, all can use the cache
```

---

## Best Practices

### ✓ DO:

1. **Use absolute paths** (starting with `/`)
   ```bash
   export OPENPI_DATA_HOME=/mnt/storage/openpi  # Good
   ```

2. **Ensure the directory exists**
   ```bash
   mkdir -p $OPENPI_DATA_HOME
   ```

3. **Use persistent configuration** for regular work
   - Add to `~/.bashrc` or `~/.zshrc`

4. **Document your setup** on team wikis or README files

5. **Use the same path across projects** on one PC (for cache sharing)

### ✗ DON'T:

1. **Don't use relative paths** (they can break)
   ```bash
   export OPENPI_DATA_HOME=./cache  # Bad
   ```

2. **Don't use `~` in exports** (can cause issues in some contexts)
   ```bash
   export OPENPI_DATA_HOME=~/openpi_cache  # Risky
   export OPENPI_DATA_HOME=$HOME/openpi_cache  # Better
   ```

3. **Don't modify project code** to change the location
   - Configuration via environment variable is the intended way

4. **Don't forget to reload** the shell after editing
   ```bash
   source ~/.bashrc
   ```

---

## Reference: Environment Variable Details

### Variable Name
```
OPENPI_DATA_HOME
```

### Default Value (if not set)
```
~/.cache/openpi
```

### Code Location
- **Definition**: `src/openpi/shared/download.py`
- **Function**: `get_cache_dir()`

### Used By
- `download.maybe_download()` - All file downloads
- `download.get_cache_dir()` - Direct cache access
- Docker compose files - Container cache mounting
- Training scripts - Normalization statistics loading

---

## Quick Reference Card

```bash
# Display current setting
echo $OPENPI_DATA_HOME

# Set temporarily (current session only)
export OPENPI_DATA_HOME=/your/path

# Set permanently (bash)
echo "export OPENPI_DATA_HOME=/your/path" >> ~/.bashrc
source ~/.bashrc

# Set permanently (zsh)
echo "export OPENPI_DATA_HOME=/your/path" >> ~/.zshrc
source ~/.zshrc

# Create directory
mkdir -p $OPENPI_DATA_HOME

# View downloaded files
ls -la $OPENPI_DATA_HOME

# Clear downloads (warning: deletes all cached files)
rm -rf $OPENPI_DATA_HOME/*

# View cache directory from Python
python -c "import sys; sys.path.insert(0, 'src'); from openpi.shared import download; print(download.get_cache_dir())"
```

---

## Summary

| Task | Command |
|------|---------|
| Set up on new PC | `echo "export OPENPI_DATA_HOME=/your/path" >> ~/.bashrc && source ~/.bashrc` |
| Check current setting | `echo $OPENPI_DATA_HOME` |
| Create cache directory | `mkdir -p $OPENPI_DATA_HOME` |
| Temporary override | `export OPENPI_DATA_HOME=/temp/path` |
| Use with Docker | `docker compose up` (env var already configured) |
| Run inference | `python` (no code changes needed) |

---

**Questions?** Check the troubleshooting section above or review the download module at `src/openpi/shared/download.py`.
