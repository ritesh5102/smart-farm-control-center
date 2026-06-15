# Production Cloud Deployment Guide

This guide explains how to deploy the **Smart Farm AI/ML Control Center** on a cloud server (such as an AWS EC2 instance, DigitalOcean Droplet, Linode, or Google Cloud VM) running Ubuntu/Debian.

---

## 📋 Prerequisites
- A cloud instance running **Ubuntu 20.04/22.04 LTS** or **Debian**.
- A public IPv4 address assigned to the server.
- Optional: A domain name pointing to the public IP (needed for HTTPS setup).

---

## 🛠️ Step 1: Configure Firewall / Security Group
Before starting, log in to your cloud provider console and open the following inbound ports on your server:
- **Port 22 (SSH):** For server management.
- **Port 80 (HTTP):** Public access to the Nginx reverse proxy.
- **Port 443 (HTTPS):** Secure traffic (required if setting up SSL).

---

## 📦 Step 2: Install Docker & Docker Compose
Log in to your cloud server via SSH and install Docker:

```bash
# Update local packages
sudo apt update && sudo apt upgrade -y

# Install prerequisite packages
sudo apt install -y curl apt-transport-https ca-certificates software-properties-common

# Add Docker’s official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.github.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose v2 (integrated command)
sudo apt install -y docker-compose-plugin

# Add your user to the docker group (avoids needing sudo for docker commands)
sudo usermod -aG docker $USER
```
*Note: Log out and back in for the user group changes to take effect.*

---

## 🚀 Step 3: Deploy the Application
1. Copy your project files to the cloud instance (using Git, SCP, or SFTP):
   ```bash
   git clone <your-repo-url> /app/smart-farm
   cd /app/smart-farm
   ```

2. Make the deployment script executable and run it:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

This script will download the lightweight `nginx:alpine` image, compile your Python Flask image with Gunicorn and OpenCV system libraries, link them inside a secure Docker network, start the services in the background, and prune stale cache files.

---

## 🔒 Step 4: Secure with SSL (HTTPS) using Let's Encrypt
To encrypt dashboard traffic and protect form data, you can install a free SSL certificate from Let's Encrypt:

1. Install **Certbot** and the Nginx plugin:
   ```bash
   sudo apt install -y certbot
   ```

2. Point your domain (e.g., `farm.yourdomain.com`) to the server's public IP.
3. Modify the server config to fetch a certificate. Temporarily stop Nginx container if binding port 80:
   ```bash
   docker compose down
   sudo certbot certonly --standalone -d farm.yourdomain.com
   ```
4. Update `nginx.conf` to handle SSL connections (mapping port 443, providing paths to `/etc/letsencrypt/live/farm.yourdomain.com/fullchain.pem`).
5. Run the docker containers back up:
   ```bash
   docker compose up -d
   ```

---

## 🔍 Step 5: Troubleshooting & Monitoring

### Check Container Status
Verify that both containers (`smart_farm_web` and `smart_farm_nginx`) are running:
```bash
docker compose ps
```

### View Live Logs
Follow the console output from Gunicorn and Nginx logs:
```bash
docker compose logs -f
```

### Accessing the Database Logs
The SQLite database file `smart_farm.db` resides inside the docker volume mount. You can read database tables directly from the host if you have `sqlite3` installed:
```bash
sqlite3 smart_farm.db "SELECT * FROM crop_predictions LIMIT 5;"
```
