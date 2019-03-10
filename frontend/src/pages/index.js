import React from 'react';

import LyricList from '../components/lyricslist'
import {Container, Col} from 'react-bootstrap'

export default class IndexPage extends React.Component {
    render() {
        return (
            <Container>
                <Col>
                    <h1 align="center">Lyrics</h1>
                    <LyricList/>
                </Col>
            </Container>
        )
    }

}