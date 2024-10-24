FROM tomcat:9.0-jdk15
COPY target/StudentSurvey-1.0-SNAPSHOT.war /usr/local/tomcat/webapps/
EXPOSE 8080