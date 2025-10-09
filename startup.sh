#!/usr/bin/env sh

# Check if port 5002 is in use
if docker ps --filter "publish=5002" --format "{{.ID}}" | grep -q .; then
    echo "Port 5002 is in use. Stopping containers..."
    docker stop $(docker ps -q --filter "publish=5002") 2>/dev/null || true
    docker ps -a -q | xargs docker rm 2>/dev/null || true
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t team2f25-streamlit:latest .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Run the Docker container
    echo "Starting container..."
    docker run -d -p 5002:5002 --name team2f25 team2f25-streamlit:latest streamlit run app.py --server.port=5002 --server.address=0.0.0.0 --server.enableCORS=false --server.baseUrlPath=/team2f25
    
    # Check if container started successfully
    if [ $? -eq 0 ]; then
        echo "Container started successfully!"
        echo ""
        echo "Access your application at:"
        echo "http://localhost:5002/team2f25"
    else
        echo "Failed to start container"
    fi
else
    echo "Build failed"
fi