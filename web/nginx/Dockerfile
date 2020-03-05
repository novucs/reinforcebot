FROM node:13.8.0-slim AS react
COPY site .
RUN npm install && npm run build

FROM nginx:1.17.4-alpine
RUN rm /etc/nginx/conf.d/default.conf
COPY --from=react /build/ /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d
