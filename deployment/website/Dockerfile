# Production Dockerfile for ChainGuard Website Dashboard
FROM node:18-alpine

WORKDIR /app

# Copy pinned package definition and install
COPY package.json .
RUN npm install --production

# Copy static website source
COPY src/ ./src

EXPOSE 3000

ENV PORT=3000

# Start static HTTP server for website dashboard
CMD ["npm", "start"]
