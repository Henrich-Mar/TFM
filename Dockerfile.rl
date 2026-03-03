# Intermediate image - base for building and installing dependencies
FROM node:16.13.2-alpine3.15 AS install

# Install required tools
RUN apk add --no-cache --virtual .gyp git python3 make g++ \
  && ln -sf python3 /usr/bin/python

WORKDIR /usr/src/app

# Install dependencies first, to cache the image.
COPY ["package.json", "package-lock.json", "./"]

# Install dependencies
RUN npm ci

# Create image for application building
FROM install AS builder

# Copy sources
COPY . .

# Polling cadence for /api/waitingfor is generated at build time.
ARG WAITING_FOR_TIMEOUT=40
ENV WAITING_FOR_TIMEOUT=${WAITING_FOR_TIMEOUT}

# Run building
RUN npm run build

# Create image to prepare prod dependencies to be copied from
FROM install AS installProd

RUN npm ci --production --prefer-offline

# Target image
FROM node:16.13.2-alpine3.15

WORKDIR /usr/src/app

# Add user tfm
RUN adduser -S -D -h /usr/src/app tfm \
  && chown -R tfm:nogroup .

# Copy required files.
COPY ["package.json", "package-lock.json", "./"]

# Copy dependencies from intermediate image
COPY --from=installProd /usr/src/app/node_modules ./node_modules

# Copy built app from intermediate image
COPY --from=builder /usr/src/app/build ./build
COPY --from=builder /usr/src/app/assets ./assets

# Create db directory and set permissions
RUN mkdir -p /usr/src/app/db && \
    chown -R tfm:nogroup /usr/src/app/db

USER tfm

# Run command.
EXPOSE 8080

CMD npm run start
