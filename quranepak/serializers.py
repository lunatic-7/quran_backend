from rest_framework import serializers

class ChatSerializer(serializers.Serializer):
    query = serializers.CharField(max_length=1000)
    chat_history = serializers.ListField(
        child=serializers.CharField(max_length=1000)
    )
