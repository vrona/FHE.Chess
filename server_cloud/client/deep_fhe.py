import sys
from concrete.ml.deployment.fhe_client_server import FHEModelClient, FHEModelServer


class FHE_chess:
    def __init__(self):
        ## client
        self.source_client = "client/source"
        self.target_client = "client/target"
        
        # source
        self.fhesource_client = FHEModelClient(self.source_client, key_dir=self.source_client)
        self.fhesource_client.load()
        
        # target
        self.fhetarget_client = FHEModelClient(self.target_client , key_dir=self.target_client)
        self.fhetarget_client.load()
        
        ## server
        self.source_server = "server/model/source"
        self.target_server = "server/model/target"
        
        # source
        self.fhesource_server = FHEModelServer(path_dir=self.source_server)#, key_dir=self.source_server)
        self.fhesource_server.load()
        
        # target
        self.fhetarget_server = FHEModelServer(path_dir=self.target_server)# , key_dir=self.target_server)
        self.fhetarget_server.load()

    """
    input: float --> quantization --> encryption
    """
    def encrypt_keys(self, clear_chessboard, clear_source=None, target=False):
        
        if clear_source is not None and target:
            
            # quantized and encryption of clear target model inputs
            serial_target_evaluation_keys = self.fhetarget_client.get_serialized_evaluation_keys()
            quantized_chessboard, quantized_source = self.fhetarget_client.model.quantize_input(clear_chessboard, clear_source)
            chess_encrypt_input, source_encrypt_input = self.fhetarget_client.quantize_encrypt_serialize(quantized_chessboard, quantized_source)

            return chess_encrypt_input, source_encrypt_input, serial_target_evaluation_keys
        else:

            # quantized and encryption of clear source model input            
            quantized_chessboard = self.fhesource_client.model.quantize_input(clear_chessboard)
            serialized_evaluation_keys = self.fhesource_client.get_serialized_evaluation_keys()
            encrypted_input = self.fhesource_client.quantize_encrypt_serialize(quantized_chessboard)
            
            return encrypted_input, serialized_evaluation_keys
    
    """
    output: float <-- dequantization <-- decryption
    """
    def decrypt(self, encrypted_output, target=False):

        if target:
            decrypted_data = self.fhetarget_client.deserialize_decrypt_dequantize(encrypted_output)
            return decrypted_data
        else:
            decrypted_data = self.fhesource_client.deserialize_decrypt_dequantize(encrypted_output)
            return decrypted_data
