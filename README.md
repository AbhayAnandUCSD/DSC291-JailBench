# DSC291 JailBench

Investigating how quantization affects LLM safety alignment using [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench).

## Overview

This project measures refusal rates of Llama-2-7b-chat across different quantization levels:

- **FP16** — full precision baseline
- **INT8** — GPTQ 8-bit quantization
- **INT4** — BitsAndBytes NF4 and GPTQ 4-bit quantization

## Setup

```bash
pip install jailbreakbench transformers accelerate bitsandbytes auto-gptq tqdm pandas litellm
```

Requires a HuggingFace token with access to `meta-llama/Llama-2-7b-chat-hf`.

## Running

Run `jailbreakbench.ipynb` on a GPU instance (tested on UCSD DSMLP).

Note: FP16 requires ≥14GB VRAM. INT4 fits on 11GB GPUs (e.g., GTX 1080 Ti).

## VS Code Remote and Cluster Access Setup Guide

This guide covers how to configure seamless, password-less access to the UCSD DSMLP cluster, set up your VS Code remote development environment, and authenticate with GitHub for secure code deployments.

**Docs:** https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032269

### Step 1: Generate Your Local SSH Identity

To connect securely without entering your password for every operation, you need an SSH key pair. We use Ed25519 as it is faster and more secure than older RSA keys.

1. Open your local terminal (on your laptop, not the cluster) and run:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. Press Enter to save the key in the default location.
3. When prompted for a passphrase, you can press Enter twice to skip it (recommended for ease of use) or provide one for extra security.
4. Display your public key and copy the entire output string (starting with `ssh-ed25519`):

   - Mac/Linux: `cat ~/.ssh/id_ed25519.pub`
   - Windows (PowerShell): `cat ~\.ssh\id_ed25519.pub`

### Step 2: Register Your Laptop with the Cluster

Now, authorize your laptop to access your DSMLP account.

1. Log in to the cluster manually one last time using your standard password:

```bash
ssh USERNAME@dsmlp-login.ucsd.edu
```

2. Run `workspace --list` to verify your account is active and note your Course ID (e.g., `sp26-cs194`), which you will need for the VS Code configuration.
3. Append your copied public key to the server's authorized list (replace the placeholder text with your actual key string):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "PASTE_YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

4. Keep this terminal window open for the next step.

### Step 3: Link the Cluster to GitHub

Because GitHub deprecated password authentication, the cluster needs its own secure handshake to push and pull repository changes.

Note: Make sure you are in your home root directory (`~`) when running these commands. If you are inside your project repository folder (e.g., `~/my-repo`), attempting to access or create a local `.ssh` directory will fail.

1. While logged into the cluster terminal, navigate to your root home directory and ensure the `.ssh` structure is correctly set up:

```bash
cd ~
mkdir -p .ssh
cd .ssh
```

2. Generate a secure identity key pair explicitly inside this folder:

```bash
ssh-keygen -t ed25519 -f id_ed25519 -C "your_email@example.com"
```

*(Press Enter twice to bypass the passphrase prompt).*

3. Display the newly generated public key:

```bash
cat id_ed25519.pub
```

4. Copy the entire output string starting with `ssh-ed25519` all the way to your email.
5. Open a browser, navigate to GitHub.com, and go to Settings -> SSH and GPG keys.
6. Click New SSH Key, name it something identifiable (e.g., `UCSD DSMLP Cluster`), paste your key into the field, and save.
7. Verify the connection back in your cluster terminal:

```bash
ssh -T git@github.com

```

*If prompted to continue connecting, type `yes` and hit Enter. You should see a success message containing your GitHub username:*

> `Hi USERNAME! You've successfully authenticated, but GitHub does not provide shell access.`

### Step 4: Configure VS Code Remote-SSH

With authentication handled, you can now link VS Code directly to your remote environment. Per the UCSD support documentation, connections are routed directly through the internal proxy gateway IP address to prevent runtime container errors.

1. Launch VS Code on your computer.
2. Open the Extensions tab (`Ctrl+Shift+X` or `Cmd+Shift+X`), search for Remote - SSH (by Microsoft), and click Install.
3. Click the Remote Explorer icon (the small monitor icon) on the far-left sidebar.
4. Set the top dropdown to SSH Targets, click the Gear icon next to the heading, and select your local user configuration file.
5. Append the following block to the bottom of the file (leaving any existing configurations untouched). Update `USERNAME` and `COURSE_NAME` to match your assignment details. Use the CPU profile for general coding and the GPU profile for deep learning workloads:

```text
# Standard CPU Computing Environment
Host MYCOURSE
    User USERNAME
    ProxyCommand ssh -i ~/.ssh/id_ed25519 USERNAME@128.54.65.160 /opt/launch-sh/bin/launch.sh -W MYCOURSE -H -N vscode-dsmlp

# Machine Learning Environment (Requests 1 GPU Allocation)
Host MYCOURSE
    User USERNAME
    ProxyCommand ssh -i ~/.ssh/id_ed25519 USERNAME@128.54.65.160 /opt/launch-sh/bin/launch-scipy-ml.sh -g 1 -W MYCOURSE -H -N vscode-dsmlp
```

6. Save and close the file.

### Step 5: Launch Your Environment and Pull Your Code

1. Return to the Remote Explorer sidebar in VS Code.
2. Right-click your preferred target and select Connect to Host in Current Window.
3. When prompted at the top of the screen to select the platform, choose Linux.
4. Once the connection finalizes, the bottom-left corner of VS Code will display `SSH: MYCOURSE`.
5. Open an integrated VS Code terminal. You can now seamlessly clone, pull, and push your repositories without credential roadblocks:

```bash
cd ~/my-repo
git pull
```

## Troubleshooting

### Error: `Author identity unknown` on `git commit`
If Git blocks your commit with a message asking "Please tell me who you are," it means your fresh environment container does not have your Git profile configured yet.
1. Run the following commands in the terminal (replace with your actual name and GitHub email):
   ```bash
   git config --global user.email "your_email@example.com"
   git config --global user.name "Your Name"
```

2. Once configured, re-run your `git commit` command.

### Error: `Permission denied (publickey)` on `git pull` / `git push`

If you encounter this, your cluster environment is not offering the right key to GitHub.

1. Run `ls -al ~/.ssh` on the cluster and verify `id_ed25519` and `id_ed25519.pub` exist.
2. If the files exist but you are still denied, force test the handshake using the specific file path:
```bash
ssh -i ~/.ssh/id_ed25519 -T git@github.com
```

3. If this works, add a local configuration file to enforce its usage automatically. Run `nano ~/.ssh/config` and add:
```text
Host github.com
  IdentityFile ~/.ssh/id_ed25519
```

### Error: `There is no tracking information for the current branch`

If Git indicates it does not know which remote branch to merge with during a pull, explicitly link your local branch to the remote upstream origin branch:

```bash
git branch --set-upstream-to=origin/<branch_name> <branch_name>
```