FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Build for production
RUN npm run build

# Serve using a simple static server (or Vite dev server for now)
# In production, we'd use Nginx, but for this setup we mirror the user's dev flow
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
